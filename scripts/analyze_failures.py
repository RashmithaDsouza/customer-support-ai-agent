import os
import sys
import pandas as pd
from pathlib import Path

def analyze_failures():
    baseline_path = Path("data/evaluation/baseline_predictions.csv")
    agent_path = Path("data/evaluation/agent_predictions.csv")
    judge_path = Path("data/evaluation/reply_judge_results.csv")
    csv_out = Path("data/evaluation/failure_analysis.csv")
    md_out = Path("data/evaluation/failure_analysis.md")
    
    if not all([baseline_path.exists(), agent_path.exists(), judge_path.exists()]):
        print("Error: Missing evaluation files. Run run_evaluation.py first.")
        return
        
    baseline_df = pd.read_csv(baseline_path)
    agent_df = pd.read_csv(agent_path)
    judge_df = pd.read_csv(judge_path)
    
    merged = pd.merge(baseline_df, agent_df, on="message_id", suffixes=("", "_agent"))
    merged = pd.merge(merged, judge_df.drop(columns=["customer_text", "reply"]), on="message_id")
    
    failures = []
    
    for idx, row in merged.iterrows():
        # 1. Intent Misclassification
        if row["gold_intent"] != row["agent_intent"]:
            failures.append({
                "message_id": row["message_id"],
                "failure_category": "Intent Misclassification",
                "customer_text": row["customer_text"],
                "expected": row["gold_intent"],
                "predicted": row["agent_intent"],
                "generated_reply": row["agent_reply"],
                "likely_cause": "Semantic ambiguity or lack of distinct keywords.",
                "proposed_improvement": "Expand intent definitions with more diverse examples in few-shot prompt."
            })
            
        # 2. Escalation Failure
        elif row["gold_action"] != row["agent_action"]:
            failures.append({
                "message_id": row["message_id"],
                "failure_category": "Escalation Failure",
                "customer_text": row["customer_text"],
                "expected": row["gold_action"],
                "predicted": row["agent_action"],
                "generated_reply": row["agent_reply"],
                "likely_cause": "Agent rules did not align with human intuition for escalation threshold.",
                "proposed_improvement": "Refine hybrid rule conditions for borderline intents."
            })
            
        # 3. Poor Reply / Groundedness
        elif row["overall_score"] < 4:
            failures.append({
                "message_id": row["message_id"],
                "failure_category": "Poor Reply Quality",
                "customer_text": row["customer_text"],
                "expected": "Score >= 4",
                "predicted": f"Score: {row['overall_score']}",
                "generated_reply": row["agent_reply"],
                "likely_cause": "Retrieval corpus missing relevant historical cases, or LLM hallucination.",
                "proposed_improvement": "Expand retrieval corpus and strictly enforce grounding prompt instructions."
            })
            
    fail_df = pd.DataFrame(failures)
    fail_df.to_csv(csv_out, index=False)
    
    # Generate Top 5 failures
    if len(fail_df) > 0:
        top_5 = fail_df.head(5)
    else:
        top_5 = pd.DataFrame()
        
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Failure Analysis Report\n\n")
        f.write(f"Total Failures Detected: {len(failures)}\n\n")
        
        f.write("## Top 5 Failure Modes\n\n")
        for i, row in top_5.iterrows():
            f.write(f"### Failure {i+1}: {row['failure_category']}\n")
            f.write(f"- **Message ID**: {row['message_id']}\n")
            f.write(f"- **Customer Text**: {row['customer_text']}\n")
            f.write(f"- **Expected**: {row['expected']} | **Predicted**: {row['predicted']}\n")
            f.write(f"- **Likely Cause**: {row['likely_cause']}\n")
            f.write(f"- **Proposed Improvement**: {row['proposed_improvement']}\n")
            f.write(f"- **Generated Reply**:\n> {row.get('generated_reply', 'N/A')}\n\n")
            
    print(f"Failure analysis saved to {csv_out} and {md_out}")

if __name__ == "__main__":
    analyze_failures()
