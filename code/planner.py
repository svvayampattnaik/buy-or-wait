"""
planner.py — Plan generation, safety evaluation, ranking, and output formatting.
"""

from datetime import datetime, timedelta

from forecaster import (
    detect_recurring_series,
    apply_series_modifications,
    project_events,
    simulate_balance,
    compute_amount_safe_to_pay,
    compute_earliest_full_payment_date,
    infer_salary_from_scheduled,
    _parse_date,
)
from spending_changes import (
    generate_spending_change_candidates,
    select_best_spending_changes,
    format_spending_changes,
)

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Generate candidate plans
# ---------------------------------------------------------------------------

def generate_candidate_plans(request, user_profile, payment_options,
                              amount_safe_to_pay, earliest_full_date):
    """
    Generate all candidate payment plans the user might consider.
    full_payment and partial_payment are unconditional (not gated on payment_options).
    Installments must match a supplied payment option.
    """
    plans = []
    user_methods = _parse_pipe_list(
        user_profile.get("payment_methods_user_will_consider", "")
    )
    requested_amount = float(request["requested_amount"])
    request_date = request["request_date"]
    desired_date = request["desired_completion_date"]
    allows_partial = _str_to_bool(request.get("allows_partial_payment", False))
    max_months = user_profile.get("max_installment_months")
    if pd.notna(max_months):
        max_months = int(max_months)
    else:
        max_months = None

    # --- Full Payment (unconditional — does NOT need payment_options row) ---
    if "full_payment" in user_methods:
        plans.append({
            "method": "full_payment",
            "option_id": None,
            "schedule": [(request_date, requested_amount)],
            "total_paid": requested_amount,
            "completes_by": request_date,
        })

    # --- Installments (MUST match a supplied payment option) ---
    if "installments" in user_methods and max_months:
        for _, opt in payment_options.iterrows():
            if opt["payment_method"] != "installments":
                continue
            if int(opt["number_of_payments"]) > max_months:
                continue

            schedule = _generate_installment_schedule(opt)
            last_date = schedule[-1][0]
            plans.append({
                "method": "installments",
                "option_id": opt["payment_option_id"],
                "schedule": schedule,
                "total_paid": float(opt["total_payable_amount"]),
                "completes_by": last_date,
            })

    # --- Partial Payment (unconditional) ---
    if "partial_payment" in user_methods and allows_partial:
        if amount_safe_to_pay > 0 and amount_safe_to_pay < requested_amount:
            if earliest_full_date and earliest_full_date <= desired_date:
                remaining = round(requested_amount - amount_safe_to_pay, 2)
                plans.append({
                    "method": "partial_payment",
                    "option_id": None,
                    "schedule": [
                        (request_date, amount_safe_to_pay),
                        (earliest_full_date, remaining),
                    ],
                    "total_paid": requested_amount,
                    "completes_by": earliest_full_date,
                })

    # --- Wait ---
    if "full_payment" in user_methods:
        if earliest_full_date and earliest_full_date > request_date:
            plans.append({
                "method": "wait",
                "option_id": None,
                "schedule": [(earliest_full_date, requested_amount)],
                "total_paid": requested_amount,
                "completes_by": earliest_full_date,
            })

    return plans


# ---------------------------------------------------------------------------
# 2. Evaluate and rank plans
# ---------------------------------------------------------------------------

def select_best_plan(plans, recurring_series, scheduled_events, pending_debits,
                      synthetic_events, start_balance, min_balance,
                      user_profile, request, spending_change_candidates):
    """
    Evaluate each plan for 90-day safety (with and without spending changes).
    Rank by the tie-breaker cascade and return the best.
    """
    request_date = request["request_date"]
    desired_date = request["desired_completion_date"]
    evaluated = []

    for plan in plans:
        # Project events WITHOUT spending changes
        future_no_changes = project_events(
            recurring_series, scheduled_events, pending_debits,
            synthetic_events, request_date, spending_changes=None
        )

        safe_no_changes, _ = simulate_balance(
            start_balance, future_no_changes, plan["schedule"], min_balance
        )

        if safe_no_changes:
            evaluated.append((plan, None))
            continue

        # Try with spending changes
        if spending_change_candidates:
            # Compute the shortfall: how much more per period do we need?
            shortfall = _compute_per_period_shortfall(
                start_balance, future_no_changes, plan["schedule"], min_balance
            )

            if request["request_id"] == "request_06":
                print(f"\n[DEBUG] request_06")
                print(f"shortfall: {shortfall}")
                print(f"spending_change_candidates:")
                for c in spending_change_candidates:
                    print(f"  {c}")
                print(f"willing_to_reduce: {repr(user_profile.get('expense_categories_user_is_willing_to_reduce', ''))}")
                print(f"willing_to_stop: {repr(user_profile.get('expense_categories_user_is_willing_to_stop', ''))}")

            best_changes = select_best_spending_changes(
                spending_change_candidates, shortfall, future_no_changes,
                start_balance, min_balance, plan["schedule"],
                simulate_fn=simulate_balance,
                project_events_fn=project_events,
                recurring_series=recurring_series,
                scheduled_events=scheduled_events,
                pending_debits=pending_debits,
                synthetic_events=synthetic_events,
                start_date=request_date,
            )

            if best_changes:
                # Verify with full re-projection including changes
                future_with_changes = project_events(
                    recurring_series, scheduled_events, pending_debits,
                    synthetic_events, request_date, spending_changes=best_changes
                )
                safe_with_changes, _ = simulate_balance(
                    start_balance, future_with_changes, plan["schedule"], min_balance
                )
                if safe_with_changes:
                    evaluated.append((plan, best_changes))

    if not evaluated:
        return None, None

    # Rank by tie-breaker cascade
    evaluated.sort(key=lambda x: (
        0 if x[0]["completes_by"] <= desired_date else 1,       # 1. Complete by deadline
        0 if x[1] is None else 1,                               # 2. No spending changes
        x[0]["total_paid"],                                     # 3. Minimize total paid
        x[0]["schedule"][0][0],                                 # 4. Start earlier
        len(x[0]["schedule"]),                                  # 5. Fewer payments
        x[0].get("option_id") or "zzz",                        # 6. Lowest option_id
    ))

    return evaluated[0]


def _compute_per_period_shortfall(start_balance, future_events, payment_plan, min_balance):
    """
    Compute the approximate per-period shortfall — how much the balance dips
    below min_balance during the 90-day simulation.
    """
    balance = float(start_balance)
    max_deficit = 0

    from collections import defaultdict
    daily_net = defaultdict(float)

    for evt in future_events:
        amt = float(evt["amount"])
        if evt["direction"] == "debit":
            daily_net[evt["date"]] -= amt
        elif evt["direction"] == "credit":
            daily_net[evt["date"]] += amt

    for d, a in payment_plan:
        daily_net[d] -= float(a)

    for date_str in sorted(daily_net.keys()):
        balance += daily_net[date_str]
        deficit = float(min_balance) - balance
        if deficit > max_deficit:
            max_deficit = deficit

    return max_deficit


# ---------------------------------------------------------------------------
# 3. Determine affordability status
# ---------------------------------------------------------------------------

def determine_affordability_status(plan, changes, request, amount_safe_to_pay,
                                    earliest_full_date):
    """Determine the affordability_status based on the winning plan."""
    if plan is None:
        return "not_affordable"

    requested_amount = float(request["requested_amount"])
    desired_date = request["desired_completion_date"]
    method = plan["method"]

    # affordable_now: full payment today, no changes needed
    if method == "full_payment" and changes is None:
        if amount_safe_to_pay >= requested_amount:
            return "affordable_now"

    # affordable_with_plan: completed by deadline via non-wait method
    if plan["completes_by"] <= desired_date:
        if method in ("full_payment", "partial_payment", "installments"):
            return "affordable_with_plan"

    # affordable_later: wait (or full payment beyond deadline)
    if method == "wait":
        return "affordable_later"

    return "not_affordable"


# ---------------------------------------------------------------------------
# 4. Full pipeline for one request
# ---------------------------------------------------------------------------

def process_request(request, user_data, gemini_client=None):
    """
    Run the full decision pipeline for one request.
    Returns a dict with all output columns.
    """
    profile = user_data["profile"]
    events = user_data["events"]
    request_date = request["request_date"]
    requested_amount = float(request["requested_amount"])
    min_balance = float(profile["minimum_balance_to_keep"])
    start_balance = float(profile["current_available_balance"])

    # 1. Detect recurring series
    series = detect_recurring_series(events, request_date)

    # 1b. Infer salary from scheduled events if no credit series detected
    series = infer_salary_from_scheduled(events, series, request_date)

    # 2. Apply message-based modifications
    series = apply_series_modifications(
        series, user_data["series_modifications"], events, request_date
    )

    # 3. Separate scheduled and pending events
    scheduled = events[events["status"] == "scheduled"]
    pending = events[events["status"] == "pending"]

    # 4. Project events (no changes) for independent field computation
    future_no_changes = project_events(
        series, scheduled, pending,
        user_data["synthetic_events"], request_date, spending_changes=None
    )

    # 5. Compute independent output fields
    amount_safe = compute_amount_safe_to_pay(
        start_balance, future_no_changes, min_balance,
        requested_amount, request_date
    )

    earliest_full = compute_earliest_full_payment_date(
        start_balance, future_no_changes, min_balance,
        requested_amount, request_date
    )

    # 6. Generate spending-change candidates
    sc_candidates = generate_spending_change_candidates(series, profile)

    # 7. Generate candidate plans
    plans = generate_candidate_plans(
        request, profile, user_data["payment_options"],
        amount_safe, earliest_full
    )

    # 8. Select best plan
    best_plan, best_changes = select_best_plan(
        plans, series, scheduled, pending,
        user_data["synthetic_events"], start_balance, min_balance,
        profile, request, sc_candidates
    )

    # 9. Determine affordability status
    if best_plan is None:
        status = "not_affordable"
        method = "not_recommended"
        payment_plan_str = "none"
        changes_str = "none"
    else:
        status = determine_affordability_status(
            best_plan, best_changes, request, amount_safe, earliest_full
        )
        method = best_plan["method"]
        payment_plan_str = _format_payment_plan(best_plan["schedule"])
        changes_str = format_spending_changes(best_changes)

    # Special: if status is not_affordable, override method
    if status == "not_affordable":
        method = "not_recommended"
        payment_plan_str = "none"
        changes_str = "none"

    # Special: if status is affordable_later with wait method
    if status == "affordable_later" and method == "wait":
        payment_plan_str = _format_payment_plan(best_plan["schedule"])

    # 10. Format earliest_date_for_full_payment
    earliest_str = earliest_full if earliest_full else ""
    if status == "affordable_now":
        earliest_str = request_date

    # 11. Generate explanation
    explanation = _generate_explanation(
        request, profile, status, method, amount_safe,
        earliest_full, best_changes, best_plan, gemini_client
    )

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": round(amount_safe, 2),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": payment_plan_str,
        "earliest_date_for_full_payment": earliest_str,
        "spending_changes_needed": changes_str,
        "decision_explanation": explanation,
    }


# ---------------------------------------------------------------------------
# 5. Output formatting helpers
# ---------------------------------------------------------------------------

def _format_payment_plan(schedule):
    """Format payment schedule as YYYY-MM-DD:amount|... string."""
    if not schedule:
        return "none"
    parts = []
    for date_str, amount in schedule:
        # Format amount without trailing zeros where possible
        amt = float(amount)
        if amt == int(amt):
            parts.append(f"{date_str}:{int(amt)}")
        else:
            parts.append(f"{date_str}:{amt}")
    return "|".join(parts)


def _generate_explanation(request, profile, status, method, amount_safe,
                           earliest_full, changes, plan, gemini_client=None):
    """Generate a concise explanation of the recommendation."""
    requested_amount = float(request["requested_amount"])
    home_currency = profile["home_currency"]
    min_balance = float(profile["minimum_balance_to_keep"])

    # Format amounts with locale-appropriate separators
    def fmt(val):
        val = float(val)
        if val == int(val):
            return f"{int(val):,}"
        return f"{val:,.2f}"

    if status == "affordable_now":
        return (
            f"Pay {home_currency} {fmt(requested_amount)} today. "
            f"This leaves at least {home_currency} {fmt(min_balance)} "
            f"available over the next 90 days."
        )

    if status == "affordable_with_plan":
        if method == "full_payment" and changes:
            change_desc = _describe_changes(changes, home_currency)
            return (
                f"{change_desc}, then pay {home_currency} {fmt(requested_amount)} today. "
                f"This leaves at least {home_currency} {fmt(min_balance)} available."
            )
        if method == "installments" and plan:
            n = len(plan["schedule"])
            per_payment = plan["schedule"][0][1]
            start = plan["schedule"][0][0]
            return (
                f"Use {n} installments of {home_currency} {fmt(per_payment)}, "
                f"starting {start}. This leaves at least {home_currency} "
                f"{fmt(min_balance)} available."
            )
        if method == "partial_payment" and plan:
            first = plan["schedule"][0]
            second = plan["schedule"][1]
            return (
                f"Pay {home_currency} {fmt(first[1])} today and the remaining "
                f"{home_currency} {fmt(second[1])} on {second[0]}. "
                f"This completes the full request and keeps the "
                f"{home_currency} {fmt(min_balance)} minimum protected."
            )

    if status == "affordable_later":
        if earliest_full:
            return (
                f"Pay {home_currency} {fmt(requested_amount)} in full on "
                f"{earliest_full}. Paying earlier would take the balance "
                f"below the {home_currency} {fmt(min_balance)} minimum."
            )

    if status == "not_affordable":
        desired_date = request["desired_completion_date"]
        if amount_safe > 0:
            return (
                f"Do not proceed with the {home_currency} {fmt(requested_amount)} "
                f"request. Although {home_currency} {fmt(amount_safe)} is available "
                f"today, the full amount cannot be completed safely within 90 days."
            )
        return (
            f"Do not make this payment by {desired_date}. "
            f"None of the available options keeps the "
            f"{home_currency} {fmt(min_balance)} minimum protected."
        )

    return f"Decision: {status}, method: {method}."


def _describe_changes(changes, currency):
    """Generate human-readable description of spending changes."""
    if not changes:
        return ""
    parts = []
    for c in changes:
        if c["type"] == "stop":
            parts.append(f"Stop the {c.get('description', c['event_id'])}")
        elif c["type"] == "reduce_to":
            parts.append(
                f"Reduce the {c.get('description', c['event_id'])} "
                f"to {currency} {c['new_amount']:,.2f}"
            )
    return ", ".join(parts)


def _generate_installment_schedule(opt):
    """Generate (date, amount) pairs from a payment option row."""
    n = int(opt["number_of_payments"])
    amount = float(opt["payment_amount"])
    first_date = str(opt["first_payment_date"])
    freq_days = int(opt["payment_frequency_days"])

    schedule = []
    current_date = _parse_date(first_date)
    for i in range(n):
        schedule.append((current_date.strftime("%Y-%m-%d"), amount))
        current_date += timedelta(days=freq_days)

    return schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pipe_list(value):
    """Parse pipe-separated string into a set."""
    if not value or (isinstance(value, float) and pd.isna(value)):
        return set()
    return set(str(value).split("|"))


def _str_to_bool(value):
    """Convert string/bool to bool."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")
