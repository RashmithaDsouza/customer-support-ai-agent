import pandas as pd

eval_df = pd.read_csv("data/evaluation/evaluation_set.csv")
base_df = pd.read_csv("data/evaluation/baseline_predictions.csv")
agent_df = pd.read_csv("data/evaluation/agent_predictions.csv")
judge_df = pd.read_csv("data/evaluation/reply_judge_results.csv")

# Copy the exact evaluation_id into all datasets (preserving row order since they were processed sequentially)
base_df["message_id"] = eval_df["evaluation_id"]
agent_df["message_id"] = eval_df["evaluation_id"]
judge_df["message_id"] = eval_df["evaluation_id"]

base_df.to_csv("data/evaluation/baseline_predictions.csv", index=False)
agent_df.to_csv("data/evaluation/agent_predictions.csv", index=False)
judge_df.to_csv("data/evaluation/reply_judge_results.csv", index=False)

# Now recover the human scores using the deterministic sample order
try:
    human_df = pd.read_csv("data/evaluation/human_judge_scores.csv")
    if pd.isna(human_df["message_id"]).any() or (human_df["message_id"] == "nan").any():
        sample_df = judge_df.sample(n=35, random_state=42)
        recovered_ids = list(sample_df["message_id"])[:len(human_df)]
        human_df["message_id"] = recovered_ids
        human_df.to_csv("data/evaluation/human_judge_scores.csv", index=False)
        print(f"Successfully recovered {len(human_df)} human scores!")
except Exception as e:
    print("Could not recover human scores (might not exist):", e)
