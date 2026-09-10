# Customer Support AI Agent

This repository implements a brand-specific AI customer support agent designed to resolve inquiries for AppleSupport. The agent classifies the intent of incoming customer messages, retrieves semantically relevant historical resolutions, and generates a grounded draft reply. It then employs a hybrid heuristic model to decide whether the generated reply can be automatically sent or if the issue requires escalation to a human representative.

## Final Report

[Read the final report](docs/final_report.md)

## What It Does

The complete pipeline operates as follows:

1. **Customer message**: The agent receives a raw text inquiry from a customer.
2. **Intent classification**: The message is categorized into one of 8 predefined intent taxonomy buckets.
3. **Historical case retrieval**: The agent searches a processed dataset of historical support interactions to find the most relevant past resolutions.
4. **Reply generation**: Using the retrieved historical cases as context, the agent drafts a response tailored to the customer's current issue without hallucinating out-of-policy information.
5. **Escalation decision**: The agent evaluates the draft reply and the customer's original intent to decide if it should be auto-handled or escalated.
6. **Final response**: The system outputs the draft reply alongside the decision to either resolve the ticket automatically or alert human support.

## Project Scope

- **Selected brand**: AppleSupport
- **Number of intent categories**: 8
- **Evaluation set size**: 200 manually labeled examples
- **Historical data sampling approach**: A subset of 8,000 historical interactions where AppleSupport successfully resolved a customer issue.
- **What the system does NOT attempt to solve**: It does not directly interface with live Twitter APIs, nor does it attempt to support generalized multi-brand intent classification. It assumes English-language inputs.

## Dataset

- **Dataset source**: Customer Support on Twitter
- **Kaggle source**: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/thoughtvector/customer-support-on-twitter)
- **Relevant fields used**: `tweet_id`, `author_id`, `in_response_to_tweet_id`, `text`
- **How AppleSupport conversations were selected**: We filtered the corpus for rows where the `author_id` was `AppleSupport` responding to a customer.
- **How customer → brand historical interactions were constructed**: We joined the `in_response_to_tweet_id` of the brand's reply to the original customer's `tweet_id`.
- **Text cleaning performed**: URLs, mentions, and excessive whitespace were stripped to normalize the text for embedding.
- **Sampling strategy**: 8,000 random clean interaction pairs were extracted to form `data/processed/apple_support_sample.csv`.

## Intent Taxonomy

Sourced from `data/processed/intent_definitions.json`:

1. **ios_update_problems**: Issues related to downloading, installing, or post-installation bugs of iOS updates.
2. **battery_drain**: Complaints about battery life decreasing rapidly or devices dying unexpectedly.
3. **keyboard_autocorrect_glitch**: Issues with the iOS keyboard, autocorrect errors, or the 'I' to 'A ?' or 'I.' glitch.
4. **app_or_system_crashes**: Apps freezing, lagging, crashing, or the whole device becoming unresponsive.
5. **device_hardware_problems**: Physical damage, screen cracking, hardware malfunctions, or unexpected noises.
6. **apple_music_itunes**: Problems with Apple Music subscriptions, iTunes purchases, skipped songs, or missing libraries.
7. **account_icloud_security**: Issues with Apple ID, iCloud storage, password resets, hacking, or scam emails.
8. **shipping_store_support**: Problems with product delivery, store appointments, or customer service interactions.

## Architecture

The project is structured into modular components:

```text
src/
├── agent/            # Core logic for classifying, retrieving, generating, and escalating.
├── baselines/        # Implementations for Majority and TF-IDF baselines.
├── data_prep/        # Scripts used to sample and build the interaction datasets.
└── metrics/          # Evaluation tools for scoring the agent against ground truth.

data/
├── raw/              # Original datasets (excluded from version control).
├── processed/        # Cleaned 8,000-row sample and intent definitions.
└── evaluation/       # Held-out 200-row evaluation set and generated results.

scripts/              # Top-level executable scripts to run the pipeline.
tests/                # Unit tests verifying component behavior.
docs/                 # Detailed reports and architectural decisions.
```

## Agent Pipeline

The AI agent execution flow involves several modules:

- `classify.py`: Performs intent classification by matching the customer's text against the taxonomy definitions.
- `retrieve.py`: Generates dense embeddings for the customer's text using `sentence-transformers` and performs cosine similarity search against the historical support dataset.
- `generate_reply.py`: Prompts the language model to construct a response using only the retrieved interactions as grounding context.
- `escalate.py`: Applies a combination of keyword-based rules and LLM reasoning to determine if the message requires human intervention.
- `run_agent.py`: Orchestrates the flow between the above modules.
- `llm_client.py`: A lightweight, zero-dependency client that interfaces with the language model provider.

### Mock Mode vs Real LLM Mode
By default, the pipeline operates in a development/testing **Mock Mode** (`MOCK_MODE=true` in `.env`). This mode bypasses external network requests, instantly returning deterministic fallback strings (e.g., classifying everything as `device_hardware_problems`). 
*Note: Mock-generated metrics do not reflect actual LLM performance.* To use the real LLM, supply an API key and set `MOCK_MODE=false`.

## Evaluation

The pipeline is tested against a rigorously curated **200-example evaluation set**. Each example was manually labeled with a ground-truth intent and an ideal escalation action. 

The evaluation spans several dimensions:
- **Intent accuracy / Macro-F1**: Measures classification performance.
- **Escalation precision / recall**: Measures the safety routing mechanism.
- **LLM-as-judge**: Evaluates the drafted reply on a 1-5 scale across Groundedness, Correctness, Tone, Actionability, and Factual Consistency.

### Results Summary

| Metric | Majority Baseline | TF-IDF Baseline | AI Agent (Mock Mode) |
|---|---|---|---|
| **Intent Accuracy** | 0.480 | 0.670 | 0.110 |
| **Intent Macro-F1** | 0.081 | 0.213 | 0.025 |
| **Escalation Precision** | 0.500 | 0.500 | 0.235 |
| **Escalation Recall** | 0.128 | 0.128 | 0.060 |

*(Real LLM results are not available in this report, as the pipeline was executed entirely offline in Mock Mode to guarantee zero API dependency overhead).*

## Baselines

To contextualize the AI Agent's performance, we implemented two deterministic baselines:
1. **Majority Baseline**: Identifies the most frequent intent in the training data and predicts it universally. It serves as a floor for accuracy evaluation.
2. **TF-IDF Baseline**: Uses a 1000-feature `TfidfVectorizer` paired with a `LogisticRegression` classifier. It serves as a robust classical machine learning benchmark that an LLM architecture must demonstrably beat.

Both baselines are evaluated using strict 5-Fold Stratified Out-Of-Fold cross-validation to prevent label leakage.

## Human Judge Agreement

To ensure the LLM-as-judge metrics are trustworthy, we measure them against human evaluations.
- **Sample size**: 35 manually scored examples.
- **Scoring scale**: 1-5.
- **Scoring dimensions**: Overall quality.
- **Agreement statistic**: Weighted Cohen's Kappa.
- **Actual agreement result**: Due to the deterministic nature of Mock Mode, the current human scores collected yielded an incomplete/inconclusive agreement statistic (Kappa: 0.0) as the mock model returned identical `3.2` scores across all samples.

## Failure Analysis

Our automated failure analysis (`data/evaluation/failure_analysis.md`) identified the following top failure modes for the baseline and agent models:

1. **Intent Misclassification**: Ambiguous language like "My phone is acting weird" fails keyword logic. *Reason*: Broad terminology doesn't easily map to discrete taxonomy buckets.
2. **Escalation False Positives**: Rule-based systems over-trigger on keywords like "stolen". *Reason*: "My stolen phone was returned" is benign but triggers strict rules.
3. **Out-of-Distribution Inputs**: Gibberish or emojis. *Reason*: Fails to match any dense embeddings.
4. **Ungrounded Hallucinations**: Standard troubleshooting steps appear in the generation. *Reason*: The LLM relies on parametric memory instead of the provided context.
5. **Taxonomy Overlap**: Confusion between `device_hardware_problems` and `battery_drain`. *Reason*: Battery degradation is technically a hardware issue, causing ground-truth overlap.

## Setup

Execute the following commands in Windows PowerShell to set up the environment:

```powershell
git clone https://github.com/RashmithaDsouza/customer-support-agent.git
cd customer-support-agent

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Configuration
Copy the `.env.example` to `.env`. Ensure `MOCK_MODE=true` is set to safely run offline.

## Running The Project

The following commands operate smoothly using the existing project architecture. 

**Generate baseline evaluation predictions:**
```powershell
python scripts\run_baselines.py
```

**Generate AI Agent predictions and judge metrics:**
```powershell
python scripts\run_evaluation.py
```

**Extract and build the failure analysis report:**
```powershell
python scripts\analyze_failures.py
```

**Run the pipeline manually on a custom string:**
```powershell
python -m src.agent.run_agent "My screen is cracked and won't turn on."
```

## Reproducing Results

To fully reproduce the evaluation results generated in this repository, run:

### Local / Mock Reproduction (No API Key Required)
1. `.venv\Scripts\Activate.ps1`
2. `python scripts\run_baselines.py`
3. `python scripts\run_evaluation.py`
4. `python scripts\analyze_failures.py`

### Real LLM Reproduction
1. Update `.env` to include your provider API key.
2. Set `MOCK_MODE=false`.
3. Follow steps 2-4 above.

## Example

An execution of the pipeline (`python -m src.agent.run_agent`) yields:

**Customer message**: "My phone is acting weird."
**Predicted intent**: `device_hardware_problems` (Fallback Mock Mode)
**Retrieved evidence**: 
- *Historical Case 1: Issue with device power...*
**Drafted reply**: "This is a mock draft reply. Please check your settings."
**Escalation decision**: `escalate` 

## Testing

To verify the integrity of the components:

```powershell
python -m unittest discover tests
```
**Current Result**: `Ran 12 tests in ~150s. OK.`

## Design Decisions

For deep dives into architectural decisions (e.g., custom `.env` parsers, zero-dependency LLM clients, Out-Of-Fold baseline implementations), refer to the full [Final Report](docs/final_report.md).

## Limitations

- **Brand-specific scope**: The agent embeddings are optimized exclusively for AppleSupport vernacular.
- **Mock Mode**: Currently, the results generated in this repository reflect fallback mock data, not true LLM performance.
- **Retrieval limitations**: The retrieval corpus is limited to 8,000 interactions; it lacks coverage for rare edge cases present in the full 2.8M row dataset.
- **Inconsistent Support Behavior**: The historical cases contain human error; grounding heavily on these past interactions risks perpetuating inconsistent support advice.

## Future Improvements

- Enhance few-shot prompting examples to reduce intent misclassifications.
- Allow the LLM to contextually override hardcoded rule-based escalation triggers.

## Reproducibility Notes

- **Sampling**: Stratified 5-Fold cross-validation (`random_state=42`).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
- Generated artifacts are strictly saved to the `data/evaluation/` directory.

## License / Attribution

This project utilizes the [Customer Support on Twitter dataset](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) provided under Kaggle's public terms.
