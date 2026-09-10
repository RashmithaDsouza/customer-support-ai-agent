import os
import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.metrics.agreement import calculate_agreement

def run_human_labeling():
    judge_out_path = Path("data/evaluation/reply_judge_results.csv")
    human_out_path = Path("data/evaluation/human_judge_scores.csv")
    
    if not judge_out_path.exists():
        print(f"Error: {judge_out_path} not found. Run evaluation first.")
        return
        
    df = pd.read_csv(judge_out_path)
    
    # Ensure consistent 35 examples to score
    sample_df = df.sample(n=35, random_state=42).copy()
    
    # Load existing scores if safely resuming
    existing_scores = []
    if human_out_path.exists():
        try:
            existing_df = pd.read_csv(human_out_path)
            existing_scores = existing_df.to_dict("records")
        except pd.errors.EmptyDataError:
            pass
            
    scored_ids = {row["message_id"] for row in existing_scores}
    pending_df = sample_df[~sample_df["message_id"].isin(scored_ids)]
    
    if pending_df.empty:
        print(f"All {len(sample_df)} examples have already been scored!")
    else:
        print("="*50)
        print("HUMAN JUDGE AGREEMENT CLI")
        print(f"Total target: 35. Already scored: {len(scored_ids)}. Remaining: {len(pending_df)}")
        print("Rate the OVERALL quality of the AI generated reply on a scale of 1 to 5.")
        print("1 = Poor, 5 = Excellent. Type 'q' to quit early and save safely.")
        print("="*50)
        
        try:
            for idx, row in pending_df.iterrows():
                print("\n" + "-"*50)
                print(f"Message ID: {row['message_id']}")
                print(f"Customer: {row['customer_text']}")
                print(f"\nAI Reply: {row['reply']}")
                print("-" * 50)
                
                while True:
                    try:
                        val = input("Your Score (1-5) or 'q': ").strip()
                    except EOFError:
                        val = 'q'
                        
                    if val.lower() == 'q':
                        print("Exiting early...")
                        raise KeyboardInterrupt
                        
                    if val in ['1', '2', '3', '4', '5']:
                        # Save immediately
                        new_score = {
                            "message_id": row['message_id'],
                            "human_overall_score": int(val),
                            "llm_overall_score": row['overall_score']
                        }
                        existing_scores.append(new_score)
                        
                        pd.DataFrame(existing_scores).to_csv(human_out_path, index=False)
                        print("Score saved.")
                        break
                    else:
                        print("Invalid input. Enter 1, 2, 3, 4, 5, or q.")
        except (KeyboardInterrupt, EOFError):
            pass

    if not existing_scores:
        print("No scores recorded. Exiting.")
        return
        
    out_df = pd.DataFrame(existing_scores)
    
    # Calculate Agreement
    agg = calculate_agreement(out_df["human_overall_score"], out_df["llm_overall_score"])
    print("\n" + "="*40)
    print("AGREEMENT STATISTICS")
    print("="*40)
    print(f"Number of human-scored examples: {agg['n_samples']} / 35")
    print(f"Weighted Cohen's Kappa: {agg['kappa']:.3f}")
    print(f"Interpretation: {agg['interpretation']}")
    print(f"Scores saved to: {human_out_path}")

if __name__ == "__main__":
    run_human_labeling()
