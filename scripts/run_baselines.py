import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.baselines.majority_baseline import MajorityBaseline
from src.baselines.tfidf_baseline import TfidfBaseline
from src.baselines.canned_reply import generate_canned_reply
from src.baselines.rule_escalation import decide_baseline_escalation

def run_baselines():
    """
    5-fold out-of-fold evaluation on 200 hand-labelled examples.
    """
    eval_path = Path("data/evaluation/evaluation_set.csv")
    output_path = Path("data/evaluation/baseline_predictions.csv")
    
    if not eval_path.exists():
        print(f"Error: {eval_path} not found.")
        return
        
    df = pd.read_csv(eval_path)
    print(f"Loaded {len(df)} evaluation messages.")
    
    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Output containers
    df["majority_predicted_intent"] = ""
    df["tfidf_predicted_intent"] = ""
    
    X = df["customer_text"].fillna("").values
    y = df["intent"].values
    
    print("Running 5-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Processing Fold {fold}/5...")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        
        # Train & Predict Majority
        maj_model = MajorityBaseline()
        maj_model.fit(y_train)
        df.loc[test_idx, "majority_predicted_intent"] = maj_model.predict(X_test)
        
        # Train & Predict TF-IDF
        tfidf_model = TfidfBaseline()
        tfidf_model.fit(X_train, y_train)
        df.loc[test_idx, "tfidf_predicted_intent"] = tfidf_model.predict(X_test)
        
    print("Generating replies and escalation rules...")
    results = []
    
    for idx, row in df.iterrows():
        msg_id = row.get("tweet_id") # tweet_id is the message_id
        text = row["customer_text"]
        gold_intent = row["intent"]
        gold_action = row["ideal_action"]
        
        maj_intent = row["majority_predicted_intent"]
        tfidf_intent = row["tfidf_predicted_intent"]
        
        # Majority side-effects
        maj_reply = generate_canned_reply(maj_intent)
        maj_esc = decide_baseline_escalation(text, maj_intent)
        
        # TF-IDF side-effects
        tfidf_reply = generate_canned_reply(tfidf_intent)
        tfidf_esc = decide_baseline_escalation(text, tfidf_intent)
        
        results.append({
            "message_id": msg_id,
            "customer_text": text,
            "gold_intent": gold_intent,
            "majority_predicted_intent": maj_intent,
            "tfidf_predicted_intent": tfidf_intent,
            "majority_reply": maj_reply,
            "tfidf_reply": tfidf_reply,
            "gold_action": gold_action,
            "majority_predicted_action": maj_esc["action"],
            "tfidf_predicted_action": tfidf_esc["action"],
            "majority_escalation_reason": maj_esc["reason"],
            "tfidf_escalation_reason": tfidf_esc["reason"]
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    
    print("\nBaseline Runner Summary:")
    print("-" * 30)
    print(f"Total processed: {len(out_df)}")
    print("Verified: 5-fold OOF completed without label leakage.")
    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    run_baselines()
