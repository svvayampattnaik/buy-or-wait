"""
Validate accuracy claims against the 25 labeled sample requests.

Proves the match-rate numbers in README.md are real and independently
verifiable. Run from the repo root:

    python evaluation/validate.py
"""

import sys
from pathlib import Path

# Resolve paths relative to this script so it works from any cwd
SCRIPT_DIR = Path(__file__).resolve().parent          # evaluation/
REPO_ROOT = SCRIPT_DIR.parent                         # buy-or-wait/
CODE_DIR = REPO_ROOT / "code"
DATASET_DIR = REPO_ROOT / "dataset"

sys.path.insert(0, str(CODE_DIR))

import pandas as pd
from data_loader import load_all_csvs, build_user_data
from planner import process_request


def main():
    sample = pd.read_csv(DATASET_DIR / "sample_requests.csv")
    data = load_all_csvs()

    predictions = []
    for _, row in sample.iterrows():
        ud = build_user_data(row["user_id"], row["request_id"], data)
        predictions.append(process_request(row, ud))

    pred_df = pd.DataFrame(predictions)

    # Columns to validate (all 6 predictable output columns)
    columns = [
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    ]

    total = len(sample)
    header = f"{'Column':<35s} {'Match':>5s} {'Total':>5s} {'Rate':>7s}"
    print(header)
    print("-" * len(header))

    for col in columns:
        true_col = sample[col].fillna("")
        pred_col = pred_df[col].fillna("")

        if col == "amount_safe_to_pay":
            # Numeric comparison with small tolerance for float rounding
            true_num = pd.to_numeric(true_col, errors="coerce").fillna(-1)
            pred_num = pd.to_numeric(pred_col, errors="coerce").fillna(-1)
            matches = int((abs(true_num - pred_num) < 0.01).sum())
        else:
            # Exact string match
            matches = int(
                (true_col.astype(str).str.strip() == pred_col.astype(str).str.strip()).sum()
            )

        pct = 100.0 * matches / total
        print(f"{col:<35s} {matches:>5d} {total:>5d} {pct:>6.1f}%")

    print()
    print(f"Validated against {total} labeled samples in dataset/sample_requests.csv.")


if __name__ == "__main__":
    main()
    sys.exit(0)
