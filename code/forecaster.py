"""
forecaster.py — Recurring series detection, 90-day projection, balance simulation.
"""

import math
import re
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Recurring series detection
# ---------------------------------------------------------------------------

def detect_recurring_series(user_events, request_date):
    """
    Detect recurring expense/income series from historical settled events.

    Groups events by (category, description pattern, flexibility) and checks
    for consistent inter-event intervals. Returns a list of series dicts.
    """
    req_dt = _parse_date(request_date)

    # Use settled, pending, and scheduled events for pattern detection
    hist = user_events[
        user_events["status"].isin(["settled", "pending", "scheduled"])
    ].copy()

    if hist.empty:
        return []

    # Calculate earliest event date across ALL of this user's history
    all_dates = [_parse_date(d) for d in hist["event_date"] if pd.notna(d)]
    earliest_event_in_full_history = min(all_dates) if all_dates else req_dt

    # Group by category + rough description + direction + flexibility
    # For subscriptions, also group by amount (they're fixed)
    series_list = []
    grouped = _group_events_into_series(hist)

    # Track leftover events for the background rate fallback
    leftover_events = []
    successful_keys = set() # (category, description)

    for key, events in grouped.items():
        if len(events) < 2:
            leftover_events.extend(events)
            continue

        events = sorted(events, key=lambda e: e["event_date"])
        intervals = []
        for i in range(1, len(events)):
            d1 = _parse_date(events[i - 1]["event_date"])
            d2 = _parse_date(events[i]["event_date"])
            intervals.append((d2 - d1).days)

        if not intervals:
            leftover_events.extend(events)
            continue

        median_interval = statistics.median(intervals)

        # Check consistency: is this actually recurring?
        if median_interval < 3 or median_interval > 45:
            leftover_events.extend(events)
            continue

        # Allow 30% variance in intervals
        consistent = sum(
            1 for iv in intervals if abs(iv - median_interval) <= median_interval * 0.35
        )
        if consistent < len(intervals) * 0.6:
            leftover_events.extend(events)
            continue

        successful_keys.add((events[0].get("category"), events[0].get("description")))

        # Round to standard periods
        period_days = _round_to_period(median_interval)

        latest = events[-1]
        latest_date = _parse_date(latest["event_date"])

        # Compute next occurrence after request_date
        next_date = latest_date
        while next_date <= req_dt:
            next_date += timedelta(days=period_days)

        # Use latest amount for projection (variable amounts like dining)
        latest_amount = float(latest["amount"]) if pd.notna(latest["amount"]) else 0

        # Compute average amount for variable categories
        amounts = [float(e["amount"]) for e in events if pd.notna(e.get("amount"))]
        avg_amount = statistics.mean(amounts) if amounts else latest_amount

        series_list.append({
            "category": latest.get("category", ""),
            "event_type": latest.get("event_type", ""),
            "description": latest.get("description", ""),
            "direction": latest.get("direction", ""),
            "flexibility": latest.get("flexibility", "fixed"),
            "minimum_allowed_amount": latest.get("minimum_allowed_amount"),
            "latest_event_id": latest["event_id"],
            "latest_amount": latest_amount,
            "avg_amount": avg_amount,
            "period_days": period_days,
            "next_date": next_date.strftime("%Y-%m-%d"),
            "currency": latest.get("currency", ""),
        })

    # -------------------------------------------------------------------------
    # Fallback: Background Burn Rate for Irregular Fixed Expenses (DISABLED)
    # -------------------------------------------------------------------------

    # Fix 1: Filter out income series where ANY event in the same category
    # has a termination keyword (e.g. "Final employer payroll").
    # The terminating event may be in a different description group than the
    # recurring series itself, so we check all leftover + series events.
    all_credit_events = hist[hist["direction"] == "credit"]
    terminated_categories = set()
    for _, evt in all_credit_events.iterrows():
        if _is_termination_event(evt.get("description", "")):
            cat = evt.get("category", "")
            evt_date = _parse_date(evt["event_date"])
            # Only terminate if this "final" event is the chronologically
            # latest credit event in this category
            same_cat = all_credit_events[all_credit_events["category"] == cat]
            latest_in_cat = max(_parse_date(d) for d in same_cat["event_date"])
            if evt_date >= latest_in_cat:
                terminated_categories.add(cat)

    series_list = [
        s for s in series_list
        if not (s["direction"] == "credit" and
                s["category"] in terminated_categories)
    ]

    return series_list


def _group_events_into_series(hist_df):
    """Group events into candidate recurring series."""
    groups = defaultdict(list)

    for _, row in hist_df.iterrows():
        # For subscriptions: group by exact description + category
        # For expenses: group by category + direction + flexibility
        event_type = row.get("event_type", "")
        category = row.get("category", "")
        direction = row.get("direction", "")
        flexibility = row.get("flexibility", "fixed")

        if event_type == "subscription":
            key = (category, row.get("description", ""), direction, "subscription")
        elif event_type == "income":
            key = (category, row.get("description", ""), direction, "income", flexibility)
        elif event_type == "debt_payment":
            key = (category, row.get("description", ""), direction, "debt_payment")
        else:
            # Expense: group by category + description + direction + flexibility
            key = (category, row.get("description", ""), direction, flexibility, event_type)

        groups[key].append(row.to_dict())

    return groups


def _round_to_period(median_days):
    """Round a median interval to a standard period."""
    if median_days <= 9:
        return 7  # weekly
    elif median_days <= 18:
        return 14  # biweekly
    elif median_days <= 24:
        return 21  # tri-weekly
    elif median_days <= 35:
        return 30  # monthly
    else:
        return round(median_days)


# ---------------------------------------------------------------------------
# 2. Apply series modifications from messages
# ---------------------------------------------------------------------------

def apply_series_modifications(series_list, modifications, user_events, request_date):
    """
    Apply modifications extracted from messages to recurring series.
    E.g., salary amount change, contract ending, salary date change.
    """
    req_dt = _parse_date(request_date)

    for mod in modifications:
        action = mod.get("action", "")
        category = mod.get("category", "")

        if action == "stop":
            # Remove series matching this category
            effective = mod.get("effective_date")
            if effective:
                eff_dt = _parse_date(effective)
                series_list = [
                    s for s in series_list
                    if not (s["category"] == category and
                            _parse_date(s["next_date"]) >= eff_dt)
                ]
            else:
                series_list = [s for s in series_list if s["category"] != category]

        elif action == "update":
            field = mod.get("field", "")
            new_value = mod.get("new_value")
            effective = mod.get("effective_date")

            for s in series_list:
                if s["category"] != category:
                    continue
                if field == "amount" and new_value is not None:
                    if effective:
                        eff_dt = _parse_date(effective)
                        if _parse_date(s["next_date"]) >= eff_dt:
                            s["latest_amount"] = float(new_value)
                            s["avg_amount"] = float(new_value)
                    else:
                        s["latest_amount"] = float(new_value)
                        s["avg_amount"] = float(new_value)
                elif field == "settlement_date" and new_value is not None:
                    s["next_date"] = str(new_value)

        elif action == "update_next":
            # Temporary change — only for the next occurrence
            field = mod.get("field", "")
            new_value = mod.get("new_value")
            for s in series_list:
                if s["category"] != category:
                    continue
                if field == "amount" and new_value is not None:
                    s["next_occurrence_override"] = {"field": field, "value": float(new_value)}
                    # Mode-based fallback for income/salary with temporary reductions
                    if s.get("event_type") == "income" or s.get("category") == "salary":
                        amounts = [float(amt) for amt in user_events[user_events["category"] == s["category"]]["amount"] if pd.notna(amt)]
                        if amounts:
                            import statistics
                            mode_amount = statistics.mode(amounts)
                            if mode_amount != s["latest_amount"]:
                                s["latest_amount"] = mode_amount
                                s["avg_amount"] = mode_amount

        elif action == "increase_pct":
            pct = mod.get("new_value", 0)
            for s in series_list:
                if s["category"] == category:
                    s["latest_amount"] = round(s["latest_amount"] * (1 + pct / 100), 2)
                    s["avg_amount"] = round(s["avg_amount"] * (1 + pct / 100), 2)

        elif action == "ignore_pair":
            # Internal transfer — find matching debit+credit pair and remove both
            pass

    return series_list


# ---------------------------------------------------------------------------
# 3. Project events forward 90 days
# ---------------------------------------------------------------------------

def project_events(recurring_series, scheduled_events, pending_debits,
                   synthetic_events, start_date, spending_changes=None):
    """
    Build the list of future cash-flow events over 90 days from start_date.
    """
    start_dt = _parse_date(start_date)
    end_dt = start_dt + timedelta(days=90)
    future = []

    # 1. Scheduled events (already confirmed, e.g. "Next confirmed salary")
    for _, evt in scheduled_events.iterrows():
        sd = _parse_date(evt["settlement_date"])
        if start_dt <= sd <= end_dt:
            future.append({
                "date": evt["settlement_date"],
                "amount": float(evt["amount"]) if pd.notna(evt["amount"]) else 0,
                "direction": evt["direction"],
                "event_id": evt["event_id"],
                "category": evt.get("category", ""),
                "source": "scheduled",
            })

    # 2. Pending debits (reserve them)
    for _, evt in pending_debits.iterrows():
        if evt["direction"] == "debit":
            sd = _parse_date(evt["settlement_date"])
            if start_dt <= sd <= end_dt:
                future.append({
                    "date": evt["settlement_date"],
                    "amount": float(evt["amount"]) if pd.notna(evt["amount"]) else 0,
                    "direction": "debit",
                    "event_id": evt["event_id"],
                    "category": evt.get("category", ""),
                    "source": "pending",
                })
        # SKIP pending credits — do not count them

    # 3. Synthetic events from messages
    for evt in synthetic_events:
        sd = evt.get("settlement_date")
        if sd:
            sd_dt = _parse_date(sd)
            if start_dt <= sd_dt <= end_dt:
                future.append({
                    "date": sd,
                    "amount": float(evt.get("amount", 0)),
                    "direction": evt.get("direction", "debit"),
                    "event_id": evt.get("event_id", "synthetic"),
                    "category": evt.get("category", ""),
                    "source": "synthetic",
                })

    # 4. Project recurring series
    for series in recurring_series:
        next_dt = _parse_date(series["next_date"])
        period = series["period_days"]
        base_amount = series["latest_amount"]
        direction = series["direction"]
        event_id = series["latest_event_id"]
        category = series["category"]
        
        is_first = True

        while next_dt <= end_dt:
            amount = base_amount
            
            # Apply one-time override on the first occurrence
            override = series.get("next_occurrence_override")
            if is_first and override and override.get("field") == "amount":
                amount = override["value"]
                
            is_first = False
            
            # Apply spending changes
            if spending_changes:
                change = _find_change_for_series(series, spending_changes)
                if change:
                    if change["type"] == "stop":
                        next_dt += timedelta(days=period)
                        continue
                    elif change["type"] == "reduce_to":
                        amount = change["new_amount"]

            if next_dt >= start_dt:
                date_str = next_dt.strftime("%Y-%m-%d")
                # Check if this date/category/direction already has a scheduled/pending/synthetic event
                duplicate = any(
                    e["date"] == date_str and e["category"] == category and e["direction"] == direction
                    for e in future
                )
                if not duplicate:
                    future.append({
                        "date": date_str,
                        "amount": amount,
                        "direction": direction,
                        "event_id": event_id,
                        "category": category,
                        "source": "projected",
                    })

            next_dt += timedelta(days=period)
            # Reset amount for next iteration (spending change may have altered it)
            amount = series["latest_amount"]
            if spending_changes:
                change = _find_change_for_series(series, spending_changes)
                if change and change["type"] == "reduce_to":
                    amount = change["new_amount"]

    # Sort by date
    future.sort(key=lambda x: x["date"])
    return future


def _find_change_for_series(series, spending_changes):
    """Find a spending change that applies to this series."""
    for change in spending_changes:
        if change["event_id"] == series["latest_event_id"]:
            return change
    return None


# ---------------------------------------------------------------------------
# 4. Balance simulation
# ---------------------------------------------------------------------------

def simulate_balance(start_balance, future_events, payment_plan, min_balance):
    """
    Simulate balance over the 90-day window.
    Returns (is_safe, min_balance_reached).

    payment_plan: list of (date_str, amount) tuples — these are DEBITS.
    """
    balance = float(start_balance)
    min_reached = balance

    # Merge future events and payment plan into one timeline, netted per day
    from collections import defaultdict
    daily_net = defaultdict(float)

    for evt in future_events:
        amt = float(evt["amount"])
        if evt["direction"] == "debit":
            daily_net[evt["date"]] -= amt
        elif evt["direction"] == "credit":
            daily_net[evt["date"]] += amt

    for date_str, amount in payment_plan:
        daily_net[date_str] -= float(amount)

    for date_str in sorted(daily_net.keys()):
        balance += daily_net[date_str]
        min_reached = min(min_reached, balance)
        
        if balance < float(min_balance):
            return False, min_reached

    return True, min_reached


# ---------------------------------------------------------------------------
# 5. Compute amount_safe_to_pay (binary search)
# ---------------------------------------------------------------------------

def compute_amount_safe_to_pay(start_balance, future_events, min_balance,
                                requested_amount, request_date):
    """
    Binary search for the largest amount payable on request_date that keeps
    the 90-day balance >= min_balance, WITHOUT any spending changes.
    """
    lo_cents = 0
    hi_cents = int(float(requested_amount) * 100)

    while lo_cents <= hi_cents:
        mid_cents = (lo_cents + hi_cents) // 2
        test_amount = mid_cents / 100.0

        payment_plan = [(request_date, test_amount)]
        safe, _ = simulate_balance(start_balance, future_events, payment_plan, min_balance)

        if safe:
            lo_cents = mid_cents + 1
        else:
            hi_cents = mid_cents - 1

    result = hi_cents / 100.0
    return max(0, min(result, float(requested_amount)))


# ---------------------------------------------------------------------------
# 6. Compute earliest_date_for_full_payment
# ---------------------------------------------------------------------------

def compute_earliest_full_payment_date(start_balance, future_events, min_balance,
                                        requested_amount, request_date):
    """
    Linear scan from request_date forward (up to 90 days) to find the first
    date when the full requested_amount passes the safety check.
    No spending changes applied.
    """
    start_dt = _parse_date(request_date)
    end_dt = start_dt + timedelta(days=90)

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")

        payment_plan = [(date_str, float(requested_amount))]
        safe, _ = simulate_balance(start_balance, future_events, payment_plan, min_balance)

        if safe:
            return date_str

        current_dt += timedelta(days=1)

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str):
    """Parse YYYY-MM-DD string to datetime."""
    if isinstance(date_str, datetime):
        return date_str
    return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")


def _is_termination_event(description):
    """Check if an event description indicates it's the final occurrence."""
    return bool(re.search(
        r'\b(final|last|termination|ending|concluded|closing)\b',
        str(description), re.IGNORECASE
    ))


def infer_salary_from_scheduled(user_events, series_list, request_date):
    """
    Fix 3: If no credit/income series was detected but a scheduled salary
    event exists, create a synthetic recurring series for it.
    This handles new employees with only one confirmed salary.
    """
    credit_series = [s for s in series_list if s["direction"] == "credit"]
    if credit_series:
        return series_list  # Already have income projection

    scheduled_income = user_events[
        (user_events["status"] == "scheduled") &
        (user_events["direction"] == "credit") &
        (user_events["category"] == "salary")
    ]

    if scheduled_income.empty:
        return series_list

    for _, evt in scheduled_income.iterrows():
        series_list.append({
            "category": evt.get("category", "salary"),
            "event_type": evt.get("event_type", "income"),
            "description": evt.get("description", ""),
            "direction": "credit",
            "flexibility": "fixed",
            "minimum_allowed_amount": None,
            "latest_event_id": evt["event_id"],
            "latest_amount": float(evt["amount"]) if pd.notna(evt["amount"]) else 0,
            "avg_amount": float(evt["amount"]) if pd.notna(evt["amount"]) else 0,
            "period_days": 30,
            "next_date": str(evt["settlement_date"])[:10],
            "currency": evt.get("currency", ""),
        })

    return series_list
