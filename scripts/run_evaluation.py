import os
import sys
import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agent.classify import classify_intent
from src.agent.retrieve import retrieve_top_k
from src.agent.generate_reply import generate_reply
from src.agent.escalate import decide_escalation
from src.metrics.evaluate import calculate_intent_metrics, calculate_escalation_metrics, judge_reply

def run_evaluation():
    baseline_path = Path("data/evaluation/baseline_predictions.csv")
    judge_out_path = Path("data/evaluation/reply_judge_results.csv")
    agent_out_path = Path("data/evaluation/agent_predictions.csv")
    intents_path = Path("data/processed/intent_definitions.json")
    
    if not baseline_path.exists():
        print(f"Error: {baseline_path} not found. Run baselines first.")
        return
        
    df = pd.read_csv(baseline_path)
    with open(intents_path, "r", encoding="utf-8") as f:
        valid_intents = [i["name"] for i in json.load(f)]
        
    print(f"Loaded {len(df)} evaluation messages from baselines.")
    
    # 1. Generate Agent Predictions (in mock mode this is instant, in real mode ~5 mins)
    print("Generating AI Agent predictions...")
    agent_results = []
    judge_results = []
    
    for idx, row in df.iterrows():
        msg_id = row["message_id"]
        text = row["customer_text"]
        gold_intent = row["gold_intent"]
        gold_action = row["gold_action"]
        
        # Agent pipeline
        class_res = classify_intent(text)
        agent_intent = class_res["intent"]
        
        cases = retrieve_top_k(text, k=3)
        retrieved_text = "\n".join([f"C: {c['customer_text']} | A: {c['reply_text']}" for c in cases])
        
        agent_reply = generate_reply(text, cases)
        
        esc_res = decide_escalation(text, agent_intent, agent_reply)
        agent_action = esc_res["action"]
        
        agent_results.append({
            "message_id": msg_id,
            "agent_intent": agent_intent,
            "agent_action": agent_action,
            "agent_reply": agent_reply
        })
        
        # LLM Judge
        j_score = judge_reply(text, retrieved_text, agent_reply)
        judge_results.append({
            "message_id": msg_id,
            "customer_text": text,
            "reply": agent_reply,
            "groundedness": j_score.get("groundedness", 1),
            "correctness": j_score.get("correctness", 1),
            "tone": j_score.get("tone", 1),
            "actionability": j_score.get("actionability", 1),
            "factual_consistency": j_score.get("factual_consistency", 1),
            "overall_score": j_score.get("overall_score", 1),
            "judge_reason": j_score.get("judge_reason", "")
        })
        
    agent_df = pd.DataFrame(agent_results)
    agent_df.to_csv(agent_out_path, index=False)
    
    judge_df = pd.DataFrame(judge_results)
    judge_df.to_csv(judge_out_path, index=False)
    
    # Merge for metrics
    merged = pd.merge(df, agent_df, on="message_id")
    
    # 2. Compute Intent Metrics
    print("\n" + "="*40)
    print("INTENT CLASSIFICATION METRICS")
    print("="*40)
    
    maj_intent_metrics = calculate_intent_metrics(merged["gold_intent"], merged["majority_predicted_intent"], valid_intents)
    tfidf_intent_metrics = calculate_intent_metrics(merged["gold_intent"], merged["tfidf_predicted_intent"], valid_intents)
    agent_intent_metrics = calculate_intent_metrics(merged["gold_intent"], merged["agent_intent"], valid_intents)
    
    print(f"Majority Accuracy: {maj_intent_metrics['accuracy']:.3f} | Macro-F1: {maj_intent_metrics['macro_f1']:.3f}")
    print(f"TF-IDF   Accuracy: {tfidf_intent_metrics['accuracy']:.3f} | Macro-F1: {tfidf_intent_metrics['macro_f1']:.3f}")
    print(f"AI Agent Accuracy: {agent_intent_metrics['accuracy']:.3f} | Macro-F1: {agent_intent_metrics['macro_f1']:.3f}")
    
    # 3. Compute Escalation Metrics
    print("\n" + "="*40)
    print("ESCALATION METRICS (vs Ideal Action)")
    print("="*40)
    
    maj_esc_metrics = calculate_escalation_metrics(merged["gold_action"], merged["majority_predicted_action"])
    tfidf_esc_metrics = calculate_escalation_metrics(merged["gold_action"], merged["tfidf_predicted_action"])
    agent_esc_metrics = calculate_escalation_metrics(merged["gold_action"], merged["agent_action"])
    
    print("Majority Baseline:")
    print(f"  Precision: {maj_esc_metrics['metrics']['escalate']['precision']:.3f} | Recall: {maj_esc_metrics['metrics']['escalate']['recall']:.3f}")
    print("TF-IDF Baseline:")
    print(f"  Precision: {tfidf_esc_metrics['metrics']['escalate']['precision']:.3f} | Recall: {tfidf_esc_metrics['metrics']['escalate']['recall']:.3f}")
    print("AI Agent:")
    print(f"  Precision: {agent_esc_metrics['metrics']['escalate']['precision']:.3f} | Recall: {agent_esc_metrics['metrics']['escalate']['recall']:.3f}")
    
    # 4. Reply Judge Metrics
    print("\n" + "="*40)
    print("LLM REPLY JUDGE SCORES")
    print("="*40)
    print(f"Average Groundedness: {judge_df['groundedness'].mean():.2f}/5.0")
    print(f"Average Correctness: {judge_df['correctness'].mean():.2f}/5.0")
    print(f"Average Tone: {judge_df['tone'].mean():.2f}/5.0")
    print(f"Average Actionability: {judge_df['actionability'].mean():.2f}/5.0")
    print(f"Average Factual Consistency: {judge_df['factual_consistency'].mean():.2f}/5.0")
    print(f"Average Overall Score: {judge_df['overall_score'].mean():.2f}/5.0")
    print("\nResults saved to:")
    print(f"- {agent_out_path}")
    print(f"- {judge_out_path}")

if __name__ == "__main__":
    run_evaluation()
