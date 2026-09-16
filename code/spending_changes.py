"""
spending_changes.py — Generate and rank spending-change candidates.

Uses the overshoot-minimization criterion verified from sample request_21:
    minimize (total_freed − shortfall)
Then prefer reduce over stop, then fewer changes.
"""

from itertools import combinations


def generate_spending_change_candidates(recurring_series, user_profile):
    """
    Generate all valid individual spending-change candidates from recurring series.
    Each candidate is a dict: {type, event_id, amount_freed, new_amount (for reduce)}.

    Only events in categories the user allows can be changed.
    reducible_or_stoppable events generate BOTH a reduce and a stop candidate.
    """
    willing_to_reduce = _parse_pipe_list(
        user_profile.get("expense_categories_user_is_willing_to_reduce", "")
    )
    willing_to_stop = _parse_pipe_list(
        user_profile.get("expense_categories_user_is_willing_to_stop", "")
    )

    candidates = []

    for series in recurring_series:
        category = series["category"]
        flex = series.get("flexibility", "fixed")
        event_id = series["latest_event_id"]
        amount = series["latest_amount"]
        min_amount = series.get("minimum_allowed_amount")

        if flex == "fixed":
            continue

        # Only debits can be stopped/reduced
        if series["direction"] != "debit":
            continue

        can_stop = category in willing_to_stop
        can_reduce = category in willing_to_reduce

        if flex == "stoppable":
            if can_stop:
                candidates.append({
                    "type": "stop",
                    "event_id": event_id,
                    "category": category,
                    "amount_freed": amount,
                    "new_amount": 0,
                })

        elif flex == "reducible":
            if can_reduce and min_amount is not None and not _is_nan(min_amount):
                freed = amount - float(min_amount)
                if freed > 0:
                    candidates.append({
                        "type": "reduce_to",
                        "event_id": event_id,
                        "category": category,
                        "amount_freed": freed,
                        "new_amount": float(min_amount),
                    })

        elif flex == "reducible_or_stoppable":
            # Generate BOTH candidates — the combo selector will pick
            if can_reduce and min_amount is not None and not _is_nan(min_amount):
                freed = amount - float(min_amount)
                if freed > 0:
                    candidates.append({
                        "type": "reduce_to",
                        "event_id": event_id,
                        "category": category,
                        "amount_freed": freed,
                        "new_amount": float(min_amount),
                    })
            if can_stop:
                candidates.append({
                    "type": "stop",
                    "event_id": event_id,
                    "category": category,
                    "amount_freed": amount,
                    "new_amount": 0,
                })

    return candidates


def select_best_spending_changes(candidates, shortfall, future_events,
                                  start_balance, min_balance, payment_plan,
                                  simulate_fn, project_events_fn,
                                  recurring_series, scheduled_events,
                                  pending_debits, synthetic_events, start_date):
    """
    Enumerate valid combinations of 1-3 changes that cover the shortfall.
    Rank by overshoot minimization, then prefer reduce over stop, then fewer changes.

    simulate_fn: the simulate_balance function from forecaster.py
    project_events_fn: the project_events function from forecaster.py
    """
    if not candidates or shortfall <= 0:
        return None

    valid_combos = []

    # Try single candidates first, then pairs, then triples
    for r in range(1, min(4, len(candidates) + 1)):
        for combo in combinations(candidates, r):
            # Constraint: same event_id cannot be both stopped AND reduced
            event_ids_by_type = {}
            conflict = False
            for c in combo:
                eid = c["event_id"]
                if eid in event_ids_by_type and event_ids_by_type[eid] != c["type"]:
                    conflict = True
                    break
                event_ids_by_type[eid] = c["type"]
            if conflict:
                continue

            # Also skip if same event appears twice with same type
            seen = set()
            dup = False
            for c in combo:
                key = (c["event_id"], c["type"])
                if key in seen:
                    dup = True
                    break
                seen.add(key)
            if dup:
                continue

            total_freed = sum(c["amount_freed"] for c in combo)
            # Pre-filter removed; simulate all valid combos

            # Verify with full simulation
            changes_list = [
                {"type": c["type"], "event_id": c["event_id"],
                 "new_amount": c["new_amount"]}
                for c in combo
            ]
            projected = project_events_fn(
                recurring_series, scheduled_events, pending_debits, synthetic_events,
                start_date, spending_changes=changes_list,
            )
            safe, _ = simulate_fn(start_balance, projected, payment_plan, min_balance)
            if not safe:
                continue

            overshoot = total_freed - shortfall
            n_stops = sum(1 for c in combo if c["type"] == "stop")

            valid_combos.append({
                "combo": list(combo),
                "changes_list": changes_list,
                "overshoot": overshoot,
                "n_stops": n_stops,
                "n_changes": len(combo),
                "total_freed": total_freed,
            })

    if not valid_combos:
        return None

    # Rank: minimize overshoot, then fewer stops, then fewer changes
    valid_combos.sort(key=lambda x: (
        x["overshoot"],
        x["n_stops"],
        x["n_changes"],
    ))

    best = valid_combos[0]
    return best["changes_list"]


def format_spending_changes(changes):
    """Format spending changes for output CSV."""
    if not changes:
        return "none"
    parts = []
    for c in changes:
        if c["type"] == "stop":
            parts.append(f"stop:{c['event_id']}")
        elif c["type"] == "reduce_to":
            parts.append(f"reduce_to:{c['event_id']}:{c['new_amount']}")
    return "|".join(parts)


def _parse_pipe_list(value):
    """Parse pipe-separated string into a set."""
    if not value or (isinstance(value, float) and _is_nan(value)):
        return set()
    return set(str(value).split("|"))


def _is_nan(value):
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
