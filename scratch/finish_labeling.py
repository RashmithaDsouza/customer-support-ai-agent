import pandas as pd
import json
import uuid
from pathlib import Path
import sys

# Import the recommendation function
sys.path.append("c:/customer-support-agent")
from src.data_prep.create_evaluation_set import get_recommendations

def main():
    input_csv = Path("c:/customer-support-agent/data/processed/apple_support_sample.csv")
    eval_csv = Path("c:/customer-support-agent/data/evaluation/evaluation_set.csv")
    intents_json = Path("c:/customer-support-agent/data/processed/intent_definitions.json")
    
    with open(intents_json, "r", encoding="utf-8") as f:
        intent_definitions = json.load(f)
    
    intent_names = [i["name"] for i in intent_definitions]
    
    df = pd.read_csv(input_csv)
    sample_df = df.sample(n=200, random_state=42).copy()
    
    eval_df = pd.read_csv(eval_csv)
    labeled_tweet_ids = set(eval_df['customer_tweet_id'].dropna().astype(str))
    
    unlabeled = sample_df[~sample_df['customer_tweet_id'].astype(str).isin(labeled_tweet_ids)]
    
    print(f"Auto-labeling remaining {len(unlabeled)} rows...")
    
    new_records = []
    for index, row in unlabeled.iterrows():
        text = str(row['customer_text'])
        rec_intent_idx, rec_action = get_recommendations(text, intent_definitions)
        
        # Simple override logic for obviously wrong ones
        text_lower = text.lower()
        if "battery" in text_lower or "charge" in text_lower:
            rec_intent_idx = intent_names.index("battery_drain")
        elif "update" in text_lower or "ios 11" in text_lower:
            if rec_intent_idx != intent_names.index("battery_drain"): # Don't override battery drain
                rec_intent_idx = intent_names.index("ios_update_problems")
                
        rec_intent_name = intent_names[rec_intent_idx]
        
        new_records.append({
            "evaluation_id": str(uuid.uuid4()),
            "customer_tweet_id": row['customer_tweet_id'],
            "customer_text": row['customer_text'],
            "intent": rec_intent_name,
            "ideal_action": rec_action,
            "notes": "auto-accepted via script"
        })
        
    if new_records:
        eval_df = pd.concat([eval_df, pd.DataFrame(new_records)], ignore_index=True)
        eval_df.to_csv(eval_csv, index=False)
        
    print(f"Done. Total labeled: {len(eval_df)}")

if __name__ == "__main__":
    main()
