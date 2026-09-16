"""
data_loader.py — Load all CSVs, resolve messages/images, convert currencies.
"""

import os
import re
import json
import base64
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


# ---------------------------------------------------------------------------
# 1. Raw CSV loading
# ---------------------------------------------------------------------------

def load_all_csvs():
    """Load every dataset CSV into a dict of DataFrames."""
    data = {
        "profiles": pd.read_csv(DATASET_DIR / "financial_profiles.csv"),
        "events": pd.read_csv(DATASET_DIR / "financial_events.csv"),
        "rates": pd.read_csv(DATASET_DIR / "exchange_rates.csv"),
        "requests": pd.read_csv(DATASET_DIR / "requests.csv"),
        "sample_requests": pd.read_csv(DATASET_DIR / "sample_requests.csv"),
        "payment_options": pd.read_csv(DATASET_DIR / "request_payment_options.csv"),
        "messages": pd.read_csv(DATASET_DIR / "messages.csv"),
        "images": pd.read_csv(DATASET_DIR / "images.csv"),
        "output_template": pd.read_csv(DATASET_DIR / "output.csv"),
    }
    # Normalise date columns to strings (YYYY-MM-DD)
    for df_name, date_cols in [
        ("events", ["event_date", "settlement_date"]),
        ("rates", ["rate_date"]),
        ("requests", ["request_date", "desired_completion_date"]),
        ("sample_requests", ["request_date", "desired_completion_date"]),
        ("payment_options", ["first_payment_date"]),
    ]:
        for col in date_cols:
            if col in data[df_name].columns:
                data[df_name][col] = data[df_name][col].astype(str)
    return data


# ---------------------------------------------------------------------------
# 2. Exchange-rate conversion
# ---------------------------------------------------------------------------

def _build_rate_lookup(rates_df):
    """
    Build a nested dict: rate_lookup[rate_date][(from, to)] = rate
    Also precompute inverses.
    """
    lookup = {}
    for _, row in rates_df.iterrows():
        d = row["rate_date"]
        pair = (row["from_currency"], row["to_currency"])
        lookup.setdefault(d, {})[pair] = float(row["rate"])
        # Store inverse
        inv_pair = (row["to_currency"], row["from_currency"])
        lookup[d].setdefault(inv_pair, 1.0 / float(row["rate"]))
    return lookup


def convert_amount(amount, from_currency, to_currency, settlement_date, rate_lookup):
    """Convert *amount* from *from_currency* to *to_currency* on *settlement_date*."""
    if from_currency == to_currency or pd.isna(amount):
        return amount

    date_rates = rate_lookup.get(settlement_date, {})

    # Direct
    direct = date_rates.get((from_currency, to_currency))
    if direct is not None:
        return round(amount * direct, 2)

    # Inverse
    inverse = date_rates.get((to_currency, from_currency))
    if inverse is not None:
        return round(amount / inverse, 2)

    # Multi-hop through any intermediate
    for (fc, tc), r1 in date_rates.items():
        if fc == from_currency:
            r2 = date_rates.get((tc, to_currency))
            if r2 is not None:
                return round(amount * r1 * r2, 2)

    # Fallback: try nearest earlier date
    all_dates = sorted(rate_lookup.keys())
    for d in reversed(all_dates):
        if d < settlement_date:
            return convert_amount(amount, from_currency, to_currency, d,
                                  {d: rate_lookup[d]})

    raise ValueError(
        f"No rate for {from_currency}->{to_currency} on {settlement_date}"
    )


# ---------------------------------------------------------------------------
# 3. Event filtering and deduplication
# ---------------------------------------------------------------------------

def filter_and_dedup_events(events_df):
    """
    Filter out non-cash events and handle linked-event deduplication.
    Returns a cleaned DataFrame.
    """
    df = events_df.copy()

    # Mark events to exclude
    exclude_ids = set()

    # 1. Exclude unrealized / non-cash (investment_valuation)
    unrealized = df[df["status"] == "unrealized"]["event_id"]
    exclude_ids.update(unrealized)

    non_cash = df[df["direction"] == "non_cash"]["event_id"]
    exclude_ids.update(non_cash)

    # 2. Exclude cancelled events — BUT if a linked successor exists
    #    and is settled/scheduled, keep the successor only
    cancelled = df[df["status"] == "cancelled"]["event_id"]
    exclude_ids.update(cancelled)

    # 3. Exclude failed events — keep scheduled retries if linked
    failed = df[df["status"] == "failed"]["event_id"]
    exclude_ids.update(failed)

    # 4. Handle linked_event_id chains
    linked = df[df["linked_event_id"].notna()]
    for _, row in linked.iterrows():
        parent_id = row["linked_event_id"]
        child_id = row["event_id"]
        parent = df[df["event_id"] == parent_id]
        if parent.empty:
            continue

        parent_status = parent.iloc[0]["status"]
        child_status = row["status"]
        child_type = row["event_type"]

        # Cancelled auth → settled purchase: keep child only
        if parent_status == "cancelled" and child_status == "settled":
            exclude_ids.add(parent_id)
            exclude_ids.discard(child_id)

        # Failed → scheduled retry: keep child only
        elif parent_status == "failed" and child_status == "scheduled":
            exclude_ids.add(parent_id)
            exclude_ids.discard(child_id)

        # Charge → refund: keep both (net zero)
        elif child_type == "refund":
            exclude_ids.discard(parent_id)
            exclude_ids.discard(child_id)

        # Investment purchase → valuation: keep purchase, exclude valuation
        elif child_type == "investment_valuation":
            exclude_ids.discard(parent_id)
            exclude_ids.add(child_id)

        # Investment purchase → sale: keep both (purchase was outflow, sale is inflow)
        elif child_type == "investment_sale":
            exclude_ids.discard(parent_id)
            exclude_ids.discard(child_id)

    df = df[~df["event_id"].isin(exclude_ids)].copy()
    return df


# ---------------------------------------------------------------------------
# 4. Image-based amount extraction
# ---------------------------------------------------------------------------

def extract_amounts_from_images(events_df, images_df, gemini_client):
    """
    For events with blank amounts, extract amount from linked image using Gemini.
    Updates events_df in-place.
    """
    blank_amount_events = events_df[events_df["amount"].isna()]
    if blank_amount_events.empty:
        return events_df

    for _, evt in blank_amount_events.iterrows():
        event_id = evt["event_id"]
        img_row = images_df[images_df["related_event_id"] == event_id]
        if img_row.empty:
            continue

        image_id = img_row.iloc[0]["image_id"]
        image_path = DATASET_DIR / "media" / "images" / f"{image_id}.png"

        if not image_path.exists():
            continue

        amount = _extract_amount_from_image(image_path, evt, gemini_client)
        if amount is not None:
            events_df.loc[events_df["event_id"] == event_id, "amount"] = amount

    return events_df


def _extract_amount_from_image(image_path, event_row, gemini_client):
    """Use Gemini vision to extract monetary amount from receipt/invoice image."""
    if gemini_client is None:
        return None

    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        f"This image is a receipt, invoice, or financial document for: "
        f"{event_row['description']}.\n"
        f"The currency is {event_row['currency']}.\n"
        f"Extract the total monetary amount from this image. "
        f"Return ONLY a JSON object with a single key 'amount' and a numeric value. "
        f"Example: {{\"amount\": 1234.56}}\n"
        f"Do not include currency symbols. Return only the JSON."
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
        )
        text = response.text.strip()
        # Parse JSON from response
        match = re.search(r'\{[^}]+\}', text)
        if match:
            parsed = json.loads(match.group())
            return float(parsed["amount"])
    except Exception as e:
        print(f"  [WARN] Image extraction failed for {event_row['event_id']}: {e}")

    return None


# ---------------------------------------------------------------------------
# 5. Message-based evidence resolution
# ---------------------------------------------------------------------------

def resolve_messages(events_df, messages_df, user_id, request_id, gemini_client):
    """
    Process messages for a user/request. Returns:
      - modified events_df (patches to existing events)
      - list of synthetic events (new financial facts from messages)
      - list of series modifications (salary changes, contract endings, etc.)
    """
    user_msgs = messages_df[
        (messages_df["user_id"] == user_id) |
        (messages_df["request_id"] == request_id)
    ].sort_values("sent_at")

    synthetic_events = []
    series_modifications = []

    for _, msg in user_msgs.iterrows():
        if pd.notna(msg.get("related_event_id")):
            # CASE A: Patch an existing event
            _apply_linked_message(events_df, msg, gemini_client)
        else:
            # CASE B: Extract new financial facts
            facts = _extract_financial_facts(msg, gemini_client)
            if facts:
                for fact in facts:
                    if fact.get("type") == "synthetic_event":
                        synthetic_events.append(fact)
                    elif fact.get("type") == "series_modification":
                        series_modifications.append(fact)

    return events_df, synthetic_events, series_modifications


def _apply_linked_message(events_df, msg, gemini_client):
    """Interpret a message linked to a specific event and apply patches."""
    event_id = msg["related_event_id"]
    event_mask = events_df["event_id"] == event_id
    if not event_mask.any():
        return

    if gemini_client is None:
        # Fallback: use keyword heuristics
        _apply_linked_heuristic(events_df, event_mask, msg)
        return

    event_row = events_df[event_mask].iloc[0]
    prompt = (
        f"A message about financial event '{event_row['event_id']}' "
        f"({event_row['description']}, {event_row['event_type']}, "
        f"{event_row['amount']} {event_row['currency']}, status={event_row['status']}):\n\n"
        f'"{msg["message_text"]}"\n\n'
        f"Does this message change any facts about this event? Return JSON:\n"
        f'{{"changes": {{"amount": <new_amount_or_null>, "settlement_date": "<new_date_or_null>", '
        f'"status": "<new_status_or_null>", "should_ignore": <true_if_not_real_cash_or_false>}}}}\n'
        f"Set fields to null if the message doesn't change them."
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            changes = parsed.get("changes", {})
            if changes.get("amount") is not None:
                events_df.loc[event_mask, "amount"] = float(changes["amount"])
            if changes.get("settlement_date") is not None:
                events_df.loc[event_mask, "settlement_date"] = changes["settlement_date"]
            if changes.get("status") is not None:
                events_df.loc[event_mask, "status"] = changes["status"]
            if changes.get("should_ignore"):
                events_df.loc[event_mask, "status"] = "ignored"
    except Exception as e:
        print(f"  [WARN] Linked message resolution failed for {event_id}: {e}")
        _apply_linked_heuristic(events_df, event_mask, msg)


def _apply_linked_heuristic(events_df, event_mask, msg):
    """Keyword-based fallback for linked messages when LLM is unavailable."""
    text = str(msg.get("message_text", "")).lower()

    if "cancel" in text or "reversed" in text:
        events_df.loc[event_mask, "status"] = "cancelled"
    elif "not reached" in text or "not been credited" in text or "still pending" in text:
        # Confirm pending status — don't count pending credits
        pass
    elif "no units have been sold" in text or "no cash proceeds" in text:
        # Unrealized investment valuation — mark as non-cash
        events_df.loc[event_mask, "status"] = "unrealized"


def _extract_financial_facts(msg, gemini_client):
    """Extract financial facts from an unlinked message. Returns a list of fact dicts."""
    if gemini_client is None:
        return _extract_facts_heuristic(msg)

    prompt = (
        f"Analyze this financial message from '{msg['source_type']}' "
        f"for user '{msg['user_id']}':\n\n"
        f'"{msg["message_text"]}"\n\n'
        f"Extract ALL financial facts. For each fact, classify it as one of:\n"
        f"1. 'series_modification' — changes to recurring income/expense "
        f"(salary amount change, salary date change, contract ended, etc.)\n"
        f"2. 'synthetic_event' — a new one-time or recurring financial event "
        f"mentioned in the message that does not correspond to any existing event row\n\n"
        f"Return JSON array. For series_modification:\n"
        f'{{"type":"series_modification","category":"salary","field":"amount",'
        f'"new_value":42750000,"effective_date":"2025-08-15","action":"update"}}\n'
        f"For contract ending:\n"
        f'{{"type":"series_modification","category":"salary","action":"stop",'
        f'"effective_date":"..."}}\n'
        f"For synthetic_event:\n"
        f'{{"type":"synthetic_event","event_type":"income","description":"...",'
        f'"category":"salary","direction":"credit","amount":1234,"currency":"EUR",'
        f'"settlement_date":"2025-08-15","status":"scheduled"}}\n\n'
        f"If no actionable financial facts, return an empty array: []\n"
        f"IMPORTANT: Pending credits, unconfirmed bonuses, lottery winnings, "
        f"commissions awaiting approval — these are NOT confirmed. "
        f"Mark them with action='ignore' or omit them."
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()
        # Find JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            # Filter out ignored items
            return [f for f in facts if f.get("action") != "ignore"]
    except Exception as e:
        print(f"  [WARN] Message fact extraction failed for {msg['message_id']}: {e}")

    return _extract_facts_heuristic(msg)


def _extract_facts_heuristic(msg):
    """Keyword-based fallback for unlinked messages."""
    text = str(msg.get("message_text", ""))
    facts = []

    # Salary amount change
    salary_match = re.search(
        r'(?:salary|gaji|pay)\b.*?(?:naik menjadi|raised to|is now|is)\s+'
        r'(?:[A-Z]{3}\s+)?([0-9,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)

    # Temporary pay reduction
    temp_match = re.search(
        r'(?:temporary|reduced).*?(?:pay|salary)\b.*?'
        r'(?:[A-Z]{3}\s+)?([0-9,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )

    if temp_match:
        amount = float(temp_match.group(1).replace(",", ""))
        facts.append({
            "type": "series_modification",
            "category": "salary",
            "field": "amount",
            "new_value": amount,
            "effective_date": date_match.group(1) if date_match else None,
            "action": "update_next",
        })
    elif salary_match:
        amount = float(salary_match.group(1).replace(",", ""))
        facts.append({
            "type": "series_modification",
            "category": "salary",
            "field": "amount",
            "new_value": amount,
            "effective_date": date_match.group(1) if date_match else None,
            "action": "update",
        })

    # Contract ended / no further income
    if re.search(r'contract has ended|no.*income.*confirmed|no.*renewal', text, re.IGNORECASE):
        facts.append({
            "type": "series_modification",
            "category": "salary",
            "action": "stop",
            "effective_date": date_match.group(1) if date_match else None,
        })

    # Salary date change
    date_change = re.search(
        r'(?:expected on|salary.*?on|confirmed.*?date.*?is)\s+(\d{4}-\d{2}-\d{2})',
        text, re.IGNORECASE
    )
    if date_change and not salary_match:
        facts.append({
            "type": "series_modification",
            "category": "salary",
            "field": "settlement_date",
            "new_value": date_change.group(1),
            "action": "update",
        })

    # Rent increase
    rent_match = re.search(r'rent.*?(?:increase|naik).*?(\d+)%', text, re.IGNORECASE)
    if rent_match:
        pct = int(rent_match.group(1))
        facts.append({
            "type": "series_modification",
            "category": "rent",
            "field": "amount",
            "action": "increase_pct",
            "new_value": pct,
        })

    # Transfer between own accounts (not real expense/income)
    if re.search(r'transfer between your.*accounts', text, re.IGNORECASE):
        facts.append({
            "type": "series_modification",
            "category": "transfer",
            "action": "ignore_pair",
        })

    # Pending payouts / unconfirmed items — stop projecting uncertain income
    if re.search(r'still pending|not.*credited|awaiting|belum', text, re.IGNORECASE):
        if re.search(r'payout|earnings|withdrawable|salary', text, re.IGNORECASE):
            facts.append({
                "type": "series_modification",
                "category": "salary",
                "action": "stop",
            })

    # First salary confirmation
    first_salary = re.search(
        r'(?:first salary|gaji pertama)\b.*?(?:[A-Z]{3}\s+)?([0-9,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if first_salary:
        amount = float(first_salary.group(1).replace(",", ""))
        eff_date = date_match.group(1) if date_match else None
        facts.append({
            "type": "synthetic_event",
            "event_type": "income",
            "description": "First salary (from message)",
            "category": "salary",
            "direction": "credit",
            "amount": amount,
            "settlement_date": eff_date,
            "status": "scheduled",
        })

    # One-time arrears adjustment
    arrears_match = re.search(
        r'arrears adjustment.*?(?:[A-Z]{3}\s+)?([0-9,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if arrears_match:
        amount = float(arrears_match.group(1).replace(",", ""))
        facts.append({
            "type": "synthetic_event",
            "event_type": "income",
            "description": "Arrears adjustment (from message)",
            "category": "salary",
            "direction": "credit",
            "amount": amount,
            "settlement_date": date_match.group(1) if date_match else None,
            "status": "scheduled",
        })

    # Invoice/payout confirmation with amount and date
    invoice_match = re.search(
        r'(?:faktur|invoice|payout).*?(?:sebesar|of)\s+(?:[A-Z]{3}\s+)?([0-9,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    payout_date = re.search(
        r'(?:diperkirakan pada|estimated|expected.*?on)\s+(\d{4}-\d{2}-\d{2})',
        text, re.IGNORECASE
    )
    if invoice_match and payout_date:
        amount = float(invoice_match.group(1).replace(",", ""))
        facts.append({
            "type": "synthetic_event",
            "event_type": "income",
            "description": "Confirmed invoice/payout (from message)",
            "category": "salary",
            "direction": "credit",
            "amount": amount,
            "settlement_date": payout_date.group(1),
            "status": "scheduled",
        })

    # New recurring expense (childcare, etc.)
    new_expense = re.search(
        r'new recurring (\w+) payment',
        text, re.IGNORECASE
    )
    if new_expense:
        # We can't determine amount from text alone; flag for LLM
        facts.append({
            "type": "series_modification",
            "category": new_expense.group(1).lower(),
            "action": "new_recurring",
        })

    return facts


# ---------------------------------------------------------------------------
# 6. Build per-user data bundle
# ---------------------------------------------------------------------------

def build_user_data(user_id, request_id, data, gemini_client=None):
    """
    Build a complete, resolved data bundle for one user+request.
    Returns dict with profile, events (filtered+converted), messages evidence,
    payment options, etc.
    """
    profile = data["profiles"][data["profiles"]["user_id"] == user_id].iloc[0]
    home_currency = profile["home_currency"]

    # Get user events
    user_events = data["events"][data["events"]["user_id"] == user_id].copy()

    # Extract amounts from images for this user's blank-amount events
    user_events = extract_amounts_from_images(
        user_events, data["images"], gemini_client
    )

    # Resolve messages
    user_events, synthetic_events, series_mods = resolve_messages(
        user_events, data["messages"], user_id, request_id, gemini_client
    )

    # Filter and deduplicate
    user_events = filter_and_dedup_events(user_events)

    # Convert foreign-currency events to home currency
    rate_lookup = _build_rate_lookup(data["rates"])
    for idx, row in user_events.iterrows():
        if pd.notna(row["amount"]) and row["currency"] != home_currency:
            converted = convert_amount(
                float(row["amount"]),
                row["currency"],
                home_currency,
                row["settlement_date"],
                rate_lookup,
            )
            user_events.at[idx, "amount"] = converted
            user_events.at[idx, "currency"] = home_currency

    # Payment options for this request
    payment_options = data["payment_options"][
        data["payment_options"]["request_id"] == request_id
    ].copy()

    return {
        "profile": profile,
        "events": user_events,
        "synthetic_events": synthetic_events,
        "series_modifications": series_mods,
        "payment_options": payment_options,
        "rate_lookup": rate_lookup,
        "home_currency": home_currency,
    }
