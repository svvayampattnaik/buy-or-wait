"""
main.py — Entry point for Buy or Wait financial agent.
"""

import sys
from pathlib import Path
import pandas as pd

from data_loader import load_all_csvs, build_user_data
from planner import process_request

def run_evaluation():
    print("Running evaluation on requests.csv...")
    data = load_all_csvs()
    requests = data["requests"]
    
    predictions = []
    
    total = len(requests)
    for idx, row in requests.iterrows():
        req_id = row["request_id"]
        user_id = row["user_id"]
        
        user_data = build_user_data(user_id, req_id, data)
        result = process_request(row, user_data)
        
        predictions.append(result)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total} requests...")
            
    pred_df = pd.DataFrame(predictions)
    
    # Required output columns
    columns_to_output = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation"
    ]
    
    output_df = pred_df[columns_to_output]
    output_df.to_csv("c:/Users/USER/Desktop/buy-or-wait/dataset/output.csv", index=False)
    print("Saved to dataset/output.csv")

if __name__ == "__main__":
    run_evaluation()