# Customer Support Agent Pipeline

This repository implements an end-to-end AI customer support agent specialized for AppleSupport on Twitter. It takes raw customer messages, classifies their intent, retrieves historically relevant resolutions, drafts grounded replies, and decides whether to automatically handle or escalate the issue to a human agent.

## Quick Start & Reproduction

You can reproduce the headline evaluation results in **under 15 minutes** using the included processed subsample, without needing to download the full 2.8M-row dataset.

### Prerequisites
- Windows OS with Python 3.9+
- A `.env` file containing `MOCK_MODE=true` (or a valid API key for OpenAI/Anthropic if testing real completions)

### Reproduction Steps
```powershell
# 1. Activate the environment
.venv\Scripts\activate

# 2. (Optional) Run pipeline unit tests
python -m unittest discover tests

# 3. Generate baseline predictions (OOF 5-fold CV)
python scripts\run_baselines.py

# 4. Generate AI Agent predictions and metrics
python scripts\run_evaluation.py

# 5. Extract failure analysis
python scripts\analyze_failures.py
```

## Problem Framing
Customer support teams spend massive amounts of time drafting replies to repetitive technical issues. We built an AI agent for `AppleSupport` capable of:
- **Intent Classification**: Routing messages into 8 specific taxonomic buckets.
- **Historically Grounded Reply Drafting**: Synthesizing context from actual historical support resolutions to ensure high correctness.
- **Escalation**: Knowing its limits by automatically escalating sensitive issues (e.g. account lockouts, legal threats) directly to a human.

## Dataset & Taxonomy
- **Source**: Kaggle Customer Support on Twitter (~3M tweets).
- **Subsample**: We extracted 8,000 clean pairs of `Customer -> AppleSupport` interactions (`data/processed/apple_support_sample.csv`).
- **Evaluation Set**: 200 rows manually labeled with golden intents and ideal actions.
- **Taxonomy**: (Stored in `data/processed/intent_definitions.json`)
  1. `battery_drain`
  2. `ios_update_problems`
  3. `app_or_system_crashes`
  4. `device_hardware_problems`
  5. `account_icloud_security`
  6. `connectivity_wifi_bluetooth`
  7. `screen_display_issues`
  8. `general_inquiry_or_other`

## System Architecture
The AI agent runs as a modular, 4-stage pipeline:
1. `classify.py`: Few-shot LLM intent classification.
2. `retrieve.py`: Dense embeddings using `SentenceTransformers` to pull the Top-3 historical resolutions from our 8k subsample.
3. `generate_reply.py`: RAG-based LLM generation strictly grounded in the retrieved cases.
4. `escalate.py`: Hybrid rule-based + LLM heuristic engine to determine `auto` vs `escalate`.

## Evaluation & Baselines
We evaluate the agent against two stringent baselines using 5-Fold Stratified Out-Of-Fold (OOF) cross-validation on our 200 labeled examples:
- **Majority Baseline**: Always predicts the majority class.
- **TF-IDF Baseline**: Classical `TfidfVectorizer` + `LogisticRegression`.

For an in-depth breakdown of results, decisions, and failure modes, read the [Final Report](docs/final_report.md).
