# Evaluation Dataset

This dataset contains ground-truth labels for customer intents and the ideal support agent action. It is strictly human-created and intended for evaluating model performance. 

## Sampling Methodology
- Source: `data/processed/apple_support_sample.csv` (which is a subsample of AppleSupport conversations where `inbound == True`).
- Total Samples: Exactly 200 customer messages.
- Sampling Method: Random sampling with `random_state=42` to ensure consistent reproducibility.

## Labeling Process
- All records in this evaluation set are manually labeled by a human.
- **NO automated tools, LLMs, or classifiers** were used to generate the ground-truth labels.
- The `src/data_prep/create_evaluation_set.py` script was used as the interactive CLI to prompt the human labeler and save results iteratively.

## Schema
- `evaluation_id`: Unique identifier for the labeled record.
- `customer_tweet_id`: The ID of the original customer tweet.
- `customer_text`: The text content of the customer's message.
- `intent`: One of the 8 predefined intents.
- `ideal_action`: Either `auto` (can be handled without human intervention) or `escalate` (requires human escalation).
- `notes`: Any additional context provided by the human labeler.
