import pandas as pd
import json
import uuid
import sys
from pathlib import Path

def get_recommendations(text, intents):
    text_lower = text.lower()
    best_intent = None
    max_score = -1
    
    # Very simple keyword matching based on intents definition
    for i, intent in enumerate(intents):
        score = 0
        
        # Check against name
        name_words = intent['name'].replace('_', ' ').split()
        for w in name_words:
            if w in text_lower:
                score += 2
                
        # Check against description
        desc = intent.get('description', '').lower()
        for w in desc.split():
            if len(w) > 4 and w in text_lower:
                score += 1
                
        # Check against examples
        for ex in intent.get('examples', []):
            for w in ex.lower().split():
                if len(w) > 4 and w in text_lower:
                    score += 0.5
                    
        if score > max_score:
            max_score = score
            best_intent = i
            
    if best_intent is None:
        best_intent = 0
        
    recommended_intent_idx = best_intent
    
    # Recommend action based on some keywords and intent
    escalate_keywords = ['broken', 'mad', 'fuck', 'shit', 'fix', 'hardware', 'crack', 'stolen', 'scam', 'angry']
    recommended_action = "auto"
    for w in escalate_keywords:
        if w in text_lower:
            recommended_action = "escalate"
            break
            
    # Also escalate hardware/security
    if intents[recommended_intent_idx]['name'] in ['device_hardware_problems', 'account_icloud_security']:
        recommended_action = "escalate"
        
    return recommended_intent_idx, recommended_action

def main():
    input_csv = Path("data/processed/apple_support_sample.csv")
    eval_csv = Path("data/evaluation/evaluation_set.csv")
    intents_json = Path("data/processed/intent_definitions.json")
    
    if not intents_json.exists():
        print(f"Error: {intents_json} not found. Please run label_intents.py first.")
        sys.exit(1)
        
    with open(intents_json, "r", encoding="utf-8") as f:
        intent_definitions = json.load(f)
    
    intent_names = [i["name"] for i in intent_definitions]
    
    if not input_csv.exists():
        print(f"Error: {input_csv} not found.")
        sys.exit(1)
        
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print("Sampling 200 messages for evaluation (seed 42)...")
    sample_df = df.sample(n=200, random_state=42).copy()
    
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["evaluation_id", "customer_tweet_id", "customer_text", "intent", "ideal_action", "notes"]
    
    if eval_csv.exists():
        print(f"Found existing evaluation set at {eval_csv}. Resuming...")
        eval_df = pd.read_csv(eval_csv)
        for col in columns:
            if col not in eval_df.columns:
                eval_df[col] = None
    else:
        print("Creating new evaluation set...")
        eval_df = pd.DataFrame(columns=columns)
        
    labeled_tweet_ids = set(eval_df['customer_tweet_id'].dropna().astype(str))
    unlabeled = sample_df[~sample_df['customer_tweet_id'].astype(str).isin(labeled_tweet_ids)]
    
    if unlabeled.empty:
        print("All 200 messages have already been labeled!")
        sys.exit(0)
        
    print(f"\nFound {len(unlabeled)} messages left to label out of 200.")
    print("Fast Mode: Press [Enter] to accept both recommendations and instantly save.\n")
    
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
        
    for index, row in unlabeled.iterrows():
        text = row['customer_text']
        print("="*80)
        print(f"ID: {row['customer_tweet_id']}")
        print(f"Text: {text}")
        print("="*80)
        
        rec_intent_idx, rec_action = get_recommendations(text, intent_definitions)
        rec_intent_name = intent_names[rec_intent_idx]
        
        print("\nIntents:")
        for i, name in enumerate(intent_names, 1):
            rec_marker = " <--- (RECOMMENDED)" if i - 1 == rec_intent_idx else ""
            print(f"  {i}. {name}{rec_marker}")
        
        print(f"\nRecommended Action: {rec_action.upper()}")
        
        intent_choice = None
        action_choice = None
        notes = ""
        
        while True:
            user_in = input(f"\n[Enter]=Accept Both, [1-8]=Override Intent, [a/e]=Override Action, [q]=Quit: ").strip().lower()
            
            if user_in == 'q':
                print("Exiting...")
                sys.exit(0)
            elif user_in == '':
                # Accept both and skip notes
                intent_choice = rec_intent_name
                action_choice = rec_action
                notes = ""
                break
            elif user_in in ['a', 'e']:
                # Override action only
                intent_choice = rec_intent_name
                action_choice = 'auto' if user_in == 'a' else 'escalate'
                
                # Notes optional
                notes_in = input("Notes (Enter for none, q to quit): ").strip()
                if notes_in.lower() == 'q': sys.exit(0)
                notes = notes_in
                break
            elif user_in.isdigit():
                idx = int(user_in) - 1
                if 0 <= idx < len(intent_names):
                    intent_choice = intent_names[idx]
                    
                    # Now ask for action override or accept rec
                    while True:
                        act_in = input(f"Action [a=auto, e=escalate, Enter={rec_action.upper()}]: ").strip().lower()
                        if act_in == 'q':
                            sys.exit(0)
                        elif act_in == '':
                            action_choice = rec_action
                            break
                        elif act_in == 'a':
                            action_choice = 'auto'
                            break
                        elif act_in == 'e':
                            action_choice = 'escalate'
                            break
                        print("Invalid input. Press 'a', 'e', or Enter.")
                        
                    # Notes optional
                    notes_in = input("Notes (Enter for none, q to quit): ").strip()
                    if notes_in.lower() == 'q': sys.exit(0)
                    notes = notes_in
                    break
            
            print("Invalid input.")
            
        # Save record
        new_record = pd.DataFrame([{
            "evaluation_id": str(uuid.uuid4()),
            "customer_tweet_id": row['customer_tweet_id'],
            "customer_text": row['customer_text'],
            "intent": intent_choice,
            "ideal_action": action_choice,
            "notes": notes
        }])
        
        eval_df = pd.concat([eval_df, new_record], ignore_index=True)
        eval_df.to_csv(eval_csv, index=False)
        print(f"\n[SAVED] {intent_choice} | {action_choice} -> ({len(eval_df)}/200 completed)\n")
        
    print("\nAll 200 messages have been labeled! Congratulations!")

if __name__ == "__main__":
    main()
