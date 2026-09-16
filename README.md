# Customer Support AI Agent — 

An end-to-end AI pipeline that ingests real, unstructured Twitter customer support messages and automatically classifies intent, generates a historically grounded reply, and decides whether to escalate to a human — without inventing facts the support history doesn't support.

Built as a portfolio project demonstrating RAG, LLM-as-judge evaluation, classical ML baselines, and rigorous offline evaluation methodology.

---

## Table of Contents

- [Problem](#problem)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [What I'd Improve Next](#what-id-improve-next)

---

## Problem

Customer support at scale is genuinely hard. Incoming messages are short, noisy, and ambiguous — "My phone is acting weird" could mean a hardware failure, a software crash, or a connectivity issue. Routing that correctly, replying appropriately, and knowing when to escalate to a human are all non-trivial decisions.

Generic chatbots fail here in a predictable way: they hallucinate policies, invent troubleshooting steps, or apologize for problems that aren't the company's fault. This project solves that by grounding every generated reply strictly in historical precedent — what AppleSupport has actually said before in comparable situations.

The dataset is the [Kaggle "Customer Support on Twitter" corpus](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~2.8M rows). The project filters it down to AppleSupport-specific conversations, making this a realistic brand-specific deployment scenario rather than a generic demo.

---

## How It Works

### Plain English Pipeline

```
Customer message
      ↓
[1] Classify intent        → which of 8 categories does this fall into?
      ↓
[2] Retrieve context       → find the 3 most similar past support cases
      ↓
[3] Generate reply         → draft a response grounded only in those cases
      ↓
[4] Escalate or resolve    → should a human take over, or is this safe to send?
```

### Technical Breakdown

**1. Data Preparation (`src/data_prep/prepare_data.py`)**
- Filters the raw TWCS corpus to AppleSupport replies only
- Reconstructs customer → brand conversation pairs via tweet ID matching
- Cleans text: decodes HTML entities, strips URLs, normalises whitespace
- Samples 8,000 clean pairs (`apple_support_sample.csv`) as the retrieval corpus

**2. Intent Taxonomy Discovery (`src/data_prep/discover_intents.py`)**
- Embeds 200 sampled customer messages using `sentence-transformers/all-MiniLM-L6-v2`
- Clusters them into 8 groups using KMeans to discover the natural intent structure
- Final taxonomy (defined in `intent_definitions.json`):
  - `battery_drain`
  - `ios_update_problems`
  - `app_or_system_crashes`
  - `device_hardware_problems`
  - `account_icloud_security`
  - `connectivity_wifi_bluetooth`
  - `screen_display_issues`
  - `general_inquiry_or_other`

**3. Intent Classifier (`src/agent/classify.py`)**
- Few-shot LLM prompt that maps the customer message to exactly one of the 8 intents
- Returns `{"intent": "...", "confidence": 0.0–1.0}`
- Falls back safely on JSON parse failures

**4. Retriever (`src/agent/retrieve.py`)**
- Encodes the incoming message with `all-MiniLM-L6-v2`
- Computes cosine similarity against 8,000 pre-computed embeddings (cached as `.npy`)
- Returns the top-3 most semantically similar historical support pairs

**5. Reply Generator (`src/agent/generate_reply.py`)**
- RAG-based LLM call — the retrieved cases are injected directly into the prompt as evidence
- Strict grounding rule: the model is instructed not to reference any policy, step, or fact that isn't present in the retrieved context
- Falls back to a polite escalation message if no clear resolution is found in context

**6. Escalation Logic (`src/agent/escalate.py`)**
- Hybrid: deterministic keyword rules first, LLM reasoning for ambiguous cases
- Hard escalation triggers: `"sue"`, `"lawyer"`, `"stolen"`, `"hacked"`, `"police"`, `"fraud"`, `"scam"`, `"manager"`
- Intent-based rules: `account_icloud_security` and `device_hardware_problems` always escalate to humans
- LLM heuristic: evaluates whether the draft reply safely resolves the case or needs intervention

**7. LLM Client (`src/agent/llm_client.py`)**
- Zero-dependency client built with Python's stdlib `urllib` — no `requests` or `openai` package required
- Supports OpenAI-compatible API endpoints
- Built-in `MOCK_MODE` for fully offline execution (no API key needed to run or test)

**8. Evaluation (`scripts/run_evaluation.py`, `src/metrics/`)**
- 200 hand-labelled examples (`evaluation_set.csv`) with gold intents and ideal actions
- **5-fold stratified out-of-fold (OOF)** cross-validation — zero label leakage by design
- LLM-as-judge scoring across 5 dimensions: Groundedness, Correctness, Tone, Actionability, Factual Consistency
- Human parity measurement via `scripts/label_judge_agreement.py` (Cohen's Kappa)

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3 (stdlib-first design) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Embedding storage | NumPy `.npy` (flat-file vector cache) |
| Similarity search | `scikit-learn` (`cosine_similarity`) |
| Baselines | `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `StratifiedKFold`) |
| LLM | OpenAI-compatible API (configurable via `.env`); `gpt-4o-mini` default |
| LLM client | Custom `urllib`-based client (zero external dependencies) |
| Data | `pandas` |
| Testing | `unittest` |

> **Note:** `requirements.txt` is empty because the project was originally built in a restricted environment where `pip install` was blocked. Dependencies are listed above and must be installed manually.

---

## Key Results

Evaluated on 200 hand-labelled examples using 5-fold stratified OOF cross-validation.

### Intent Classification

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority Baseline | 0.480 | 0.081 |
| TF-IDF + Logistic Regression | **0.670** | **0.213** |
| AI Agent (Mock LLM) | 0.110 | 0.025 |

### Escalation

| Model | Precision | Recall |
|---|---|---|
| Majority Baseline | 0.500 | 0.128 |
| TF-IDF Baseline | 0.500 | 0.128 |
| AI Agent (Mock LLM) | 0.235 | 0.060 |

### LLM Judge Scores (1–5 scale, Mock Mode)

| Dimension | Score |
|---|---|
| Overall | 3.20 / 5.0 |
| Tone | 4.00 / 5.0 |
| Groundedness | 3.00 / 5.0 |

> **Important caveat:** The AI Agent's low classification accuracy (11%) is a mock mode artifact, not a reflection of real LLM performance. `MOCK_MODE=true` returns a hardcoded fallback intent regardless of input — its purpose is pipeline stability and reproducibility, not accuracy. The TF-IDF baseline at 67% accuracy for an 8-class problem is the meaningful offline comparison. Real agent performance would require an active API key.

---

## Project Structure

```
customer-support-agent/
│
├── data/
│   ├── raw/                         # Raw TWCS corpus (not committed — too large)
│   ├── processed/
│   │   ├── apple_support_pairs.csv       # All matched customer-reply pairs
│   │   ├── apple_support_sample.csv      # 8,000-pair retrieval corpus
│   │   ├── apple_support_sample_embeddings.npy  # Cached embeddings for fast retrieval
│   │   ├── intent_definitions.json       # Canonical intent names + descriptions
│   │   └── intent_clusters.csv           # KMeans clustering output
│   └── evaluation/
│       ├── evaluation_set.csv            # 200 hand-labelled gold examples
│       ├── baseline_predictions.csv      # Majority + TF-IDF predictions
│       ├── agent_predictions.csv         # AI agent predictions
│       └── reply_judge_results.csv       # LLM-as-judge scores per example
│
├── src/
│   ├── agent/
│   │   ├── run_agent.py             # Main entry point — runs the full pipeline
│   │   ├── classify.py              # LLM-based intent classifier
│   │   ├── retrieve.py              # Dense retrieval via sentence embeddings
│   │   ├── generate_reply.py        # RAG-based reply generator
│   │   ├── escalate.py              # Hybrid rule + LLM escalation decision
│   │   └── llm_client.py            # Zero-dependency OpenAI-compatible LLM client
│   ├── baselines/
│   │   ├── majority_baseline.py     # Predicts most frequent class in training fold
│   │   ├── tfidf_baseline.py        # TF-IDF + Logistic Regression classifier
│   │   ├── canned_reply.py          # Rule-based canned replies by intent
│   │   └── rule_escalation.py       # Keyword-only escalation baseline
│   ├── data_prep/
│   │   ├── prepare_data.py          # Builds customer-reply pairs from raw TWCS data
│   │   ├── discover_intents.py      # Embeds + KMeans clusters to discover intent taxonomy
│   │   ├── label_intents.py         # LLM-assisted intent labelling for evaluation set
│   │   └── create_evaluation_set.py # Builds the 200-example gold evaluation set
│   └── metrics/
│       ├── reply_judge.py           # LLM-as-judge scoring (5 dimensions)
│       ├── intent_metrics.py        # Accuracy + Macro F1 for intent classification
│       ├── escalation_metrics.py    # Precision + Recall for escalation decisions
│       ├── agreement.py             # Cohen's Kappa for human-LLM agreement
│       └── evaluate.py              # Evaluation harness
│
├── scripts/
│   ├── run_baselines.py             # Runs 5-fold OOF evaluation for both baselines
│   ├── run_evaluation.py            # Runs full agent evaluation on 200 examples
│   ├── analyze_failures.py          # Identifies and categorises failure cases
│   ├── label_judge_agreement.py     # Computes Cohen's Kappa between human and LLM judge
│   ├── inspect_dataset.py           # Exploratory data analysis on raw TWCS corpus
│   └── hotfix_nan.py                # Utility: patches NaN values in evaluation files
│
├── tests/                           # Unit tests (run with unittest discover)
├── docs/
│   └── final_report.md              # Full write-up: problem framing, methodology, results
├── cluster_output.txt               # Raw KMeans cluster inspection output
├── .env.example                     # Environment variable template
└── .cspell.json                     # Spell-checker config for technical terms
```

---

## How to Run

### Prerequisites

Install dependencies:

```bash
pip install pandas numpy scikit-learn sentence-transformers
```

### 1. Configure environment

Copy the example env file and edit it:

```bash
copy .env.example .env
```

`.env` options:

```
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here   # leave blank to run in Mock Mode
LLM_MODEL=gpt-4o-mini
MOCK_MODE=true                  # set to false to use a real LLM
```

> Set `MOCK_MODE=true` to run the full pipeline without an API key. Outputs will use hardcoded fallback responses — useful for testing pipeline integrity.

### 2. Run the agent on a single message

```bash
python src/agent/run_agent.py "My iPhone battery drains super fast after the iOS update"
```

Example output:
```
============================================================
CUSTOMER MESSAGE:
My iPhone battery drains super fast after the iOS update
============================================================

[1/4] Classifying intent...
  -> Intent: battery_drain (Confidence: 0.95)

[2/4] Retrieving historical context...
  -> Case 1 (Sim: 0.89): My battery life has been terrible since updating...
  -> Case 2 (Sim: 0.84): After updating to iOS 16 my phone dies by noon...
  -> Case 3 (Sim: 0.81): iPhone 13 battery draining way faster than before...

[3/4] Generating grounded reply...
DRAFT REPLY:
We're sorry to hear about the battery issue after the update. ...

[4/4] Deciding escalation...
  -> Action: AUTO
  -> Reason: Standard battery inquiry resolved by context.
============================================================
```

### 3. Run baseline evaluation

```bash
python scripts/run_baselines.py
```

Runs 5-fold stratified OOF cross-validation on the 200-example evaluation set. Saves predictions to `data/evaluation/baseline_predictions.csv`.

### 4. Run full agent evaluation

```bash
python scripts/run_evaluation.py
```

Evaluates the AI agent on all 200 examples. Saves predictions and LLM judge scores to `data/evaluation/`.

### 5. Analyse failure cases

```bash
python scripts/analyze_failures.py
```

### 6. Inspect the raw dataset (requires raw TWCS data)

```bash
python scripts/inspect_dataset.py
```

> The raw TWCS corpus (`data/raw/twcs.csv`) is not committed to this repo due to its size (~2.8M rows). Download it from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it at `data/raw/twcs.csv`.

### 7. Run unit tests

```bash
python -m unittest discover tests
```

---

## What I'd Improve Next

**1. Replace flat-file retrieval with a proper vector store**
The current retrieval loads and re-encodes the full 8,000-pair corpus on each call (mitigated by `.npy` caching, but still linear scan). For a production deployment — or to scale beyond 8K examples — replacing the NumPy cosine scan with a proper approximate nearest-neighbour index (e.g., FAISS or ChromaDB) would reduce retrieval latency significantly and make the system genuinely scalable.

**2. Evaluate with a real LLM and expand the gold set**
Every AI Agent metric in the current report reflects Mock Mode fallback behaviour, not actual LLM performance. The most valuable next step is running the full evaluation pipeline with an active API key to get real intent classification and reply quality numbers. Additionally, 200 labelled examples with as few as 3 instances for some minority classes makes Macro F1 statistically unreliable — a larger gold set (1,000+ examples) would produce more stable comparisons.

**3. Tighten the escalation decision boundary**
The current hybrid escalation logic produces false positives on phrases like "my stolen phone was returned" — the keyword `"stolen"` hard-fires escalation even when context is benign. Replacing the flat keyword list with an intent-aware, context-sensitive check (letting the LLM weigh keyword presence against the full message context) would reduce unnecessary escalations without sacrificing coverage on genuinely sensitive cases.

---

## Acknowledgements

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle)
- Embeddings: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Hugging Face)
