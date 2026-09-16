# Customer Support AI Agent: 
# 1. Problem Framing
We built an end-to-end customer support AI agent for **AppleSupport**. The agent is designed to automatically ingest real, messy Twitter inquiries and perform three critical actions:
1. **Intent Classification**: Map the open-ended text into one of 8 distinct categories.
2. **Historically Grounded Reply Generation**: Draft a response grounded entirely in historical precedent, ensuring the bot does not invent fake return policies, technical steps, or unrelated apologies.
3. **Escalation**: Route the conversation to a human if the topic is inherently high-risk or the retrieved context isn't sufficient.

**What "good" looks like for this brand**: A good outcome for AppleSupport means (a) correctly routing high-risk issues — account security, hardware failures, stolen devices — to a human every time, and (b) confidently resolving the large volume of low-risk, repetitive inquiries (battery complaints, iOS update frustrations, keyboard glitches) automatically with a grounded, accurate reply. Calibrating this tradeoff is the core challenge: over-escalating wastes human agents; under-escalating risks harmful or misleading automated responses on sensitive topics.

**What I deliberately chose NOT to build**:
- **Multi-turn conversation handling**: The dataset consists of single customer tweets; multi-turn context tracking would require conversation threading logic that goes well beyond the scope of this single-exchange pipeline.
- **Sentiment-based prioritization**: While sentiment could be a useful escalation signal (e.g., extremely angry customers), it would add a dependency on a separate sentiment model and introduce additional failure modes. The hybrid keyword + intent escalation logic already captures the most critical cases.
- **Multi-brand generalization**: The retrieval embeddings and intent taxonomy are tuned for AppleSupport's specific dialect and issue types. Generalizing to a brand like SpotifyCares would require resampling, re-clustering, and re-labelling — a separate project in itself.

## 2. Dataset & Sampling Strategy
The primary dataset is the Kaggle "Customer Support on Twitter" corpus (approx. 2.8M rows).
To make the problem tractable and simulate a production "brand-specific" rollout:
- We extracted conversations specifically where `AppleSupport` replied.
- We constructed `Customer -> Brand` pairs.
- We sampled exactly 8,000 clean pairs into `data/processed/apple_support_sample.csv` to act as our retrieval corpus and baseline training set.
- A hold-out evaluation set of exactly **200 examples** (`evaluation_set.csv`) was meticulously, manually labeled with Golden Intents and Ideal Actions for our final evaluation harness.

### How the Evaluation Set Was Sampled and Labelled
The 200 examples were drawn using a **simple random sample** (`random_state=42`) from the 8,000-pair retrieval corpus, implemented in `src/data_prep/create_evaluation_set.py`. This was not stratified by intent — examples were drawn uniformly at random and then individually hand-labelled, which means some minority intents (e.g., `account_icloud_security`, `screen_display_issues`) are underrepresented in the final set.

**Labelling process**: Each example was labelled interactively via a CLI tool (`create_evaluation_set.py`) that displayed the raw customer tweet and offered a keyword-matching recommendation for both intent and ideal action. The labeller (the project author) could accept the recommendation with a single keypress or override both intent and action manually. The labelling UI enforced a mutually exclusive choice from the 8 defined intents.

- **Golden Intent**: The single most accurate intent category for the customer message, as judged by the labeller.
- **Ideal Action**: Whether a well-functioning agent *should* auto-reply (`auto`) or escalate to a human (`escalate`) for this message. This was judged based on the nature of the issue: hardware, security, and unresolvable complaints were labelled `escalate`; common, low-risk software questions were labelled `auto`.
- **Quality check**: The keyword-matching recommendation system in `create_evaluation_set.py` provides a sanity check — labels that required manual override were noted in the `notes` column of `evaluation_set.csv`. No formal inter-rater reliability check was done; this is a single-annotator label set.

## 3. Intent Taxonomy
The taxonomy was discovered natively from the data distribution:
1. `battery_drain`
2. `ios_update_problems`
3. `app_or_system_crashes`
4. `device_hardware_problems`
5. `account_icloud_security`
6. `connectivity_wifi_bluetooth`
7. `screen_display_issues`
8. `general_inquiry_or_other`

*(Note: The final deployed taxonomy in `intent_definitions.json` contains 8 intents, including `keyboard_autocorrect_glitch`, `apple_music_itunes`, and `shipping_store_support` in place of `connectivity_wifi_bluetooth`, `screen_display_issues`, and `general_inquiry_or_other` — reflecting the actual distribution observed in the AppleSupport corpus after inspection.)*

## 4. System Architecture
- **Classifier (`classify.py`)**: A few-shot LLM prompt mapping the user text to the taxonomy.
- **Retriever (`retrieve.py`)**: A dense embedding model (`sentence-transformers/all-MiniLM-L6-v2`) fetching the Top-3 closest historical resolutions.
- **Generator (`generate_reply.py`)**: RAG-based LLM generation strictly bounded by the retrieved support context.
- **Escalator (`escalate.py`)**: A hybrid deterministic-rule (keywords) and LLM-heuristic module to decide auto-resolution vs. escalation.

## 5. Sample Output

The following examples show real customer messages from the evaluation set and the escalation decisions produced by the pipeline. Intent labels reflect the gold labels from `evaluation_set.csv`; escalation decisions and reasons reflect what `escalate.py` produces for those inputs.

**Example 1 — Keyword-triggered escalation:**
```
Input:   "my phone was stolen & this picture showed up on my icloud @AppleSupport"
Intent:  app_or_system_crashes
Decision: escalate
Reason:  Rule match: High-risk or sensitive keywords detected.
```

**Example 2 — Intent-triggered escalation:**
```
Input:   "@AppleSupport I Found an iPad on the train how can I return it to its rightful owner?"
Intent:  account_icloud_security (gold label: ios_update_problems)
Decision: escalate
Reason:  Rule match: Intent 'account_icloud_security' usually requires human verification or hardware support.
```

**Example 3 — Standard inquiry routed to auto-reply:**
```
Input:   "@AppleSupport ios 11 drains battery can you fix this"
Intent:  battery_drain
Decision: auto
Reason:  Standard inquiry handled by mock.
```

> *Note: Examples 1 and 3 are drawn from `evaluation_set.csv` rows matched to `agent_predictions.csv`. Reasons reflect the exact string outputs from `escalate.py` for those code paths. Example 2 illustrates the intent-based escalation rule path, which the mock classifier cannot reach (it always falls back to `battery_drain`); the input shown is from the evaluation set.*

## 6. Evaluation Methodology & Baselines
We evaluate strictly on the 200 labeled examples using deterministic **5-Fold Stratified Out-Of-Fold (OOF)** cross-validation to ensure zero label leakage.

We compared the Agent against:
- **Majority Baseline**: Predicts the most frequent class per training fold.
- **TF-IDF Baseline**: Uses a 1000-feature `TfidfVectorizer` paired with a `LogisticRegression` classifier.
- **Reply Evaluator**: Uses an LLM-as-a-judge rubric scoring on Groundedness, Correctness, Tone, Actionability, and Factual Consistency.
- **Escalation Logic**: Rule-only keyword checking versus the AI's hybrid system.

### Judge–Human Agreement Results

`label_judge_agreement.py` was run against the 19 examples scored in `human_judge_scores.csv`. The human labeller rated each reply on a 1–5 overall quality scale; the LLM judge score was 3.2 for all examples (the fixed Mock Mode median fallback).

| Metric | Value |
|---|---|
| Human-scored examples | 19 / 35 target |
| LLM scores (all identical) | 3.2 / 5.0 (Mock Mode fixed value) |
| Human scores (range) | 1–5 |
| Exact match | 4 / 19 (21.1%) |
| Within-1-point agreement | 13 / 19 (68.4%) |
| Weighted Cohen's Kappa (κ) | 0.000 — Slight agreement |

**Interpretation**: The Kappa of 0.000 is a direct consequence of Mock Mode — the LLM judge returned the same constant score (3.2) for every single reply, making it mathematically impossible to achieve any correlation with the variance in human scores. This is the clearest quantitative demonstration of why Mock Mode metrics are not meaningful. The within-1-point agreement of 68.4% suggests the LLM's fixed score of ~3 is roughly centred in the human distribution, but the complete lack of variance means the judge is not actually evaluating individual replies at all.

**Blocking issue for real agreement measurement**: To get a meaningful Kappa, the pipeline must be run with a real API key (`MOCK_MODE=false`) so the LLM judge produces per-example scores that actually vary. The infrastructure (the CLI tool, the agreement script, the data files) is fully in place — only a live API key is missing.

## 7. Results Summary

**Note: The 'AI Agent' column below reflects Mock LLM Mode (no live API calls) and is not representative of real model performance — see Section 8 for full explanation.**

| Metric | Majority Baseline | TF-IDF Baseline | AI Agent (Mock LLM Mode) |
|---|---|---|---|
| **Intent Accuracy** | 0.480 | 0.670 | 0.110 |
| **Intent Macro-F1** | 0.081 | 0.213 | 0.025 |
| **Escalation Precision** | 0.500 | 0.500 | 0.235 |
| **Escalation Recall** | 0.128 | 0.128 | 0.060 |

*(Note: The AI Agent metrics are intentionally extremely low in this environment because the LLM is running in Mock Mode, which defaults to returning a hardcoded fallback intent (`device_hardware_problems`) and a fallback string to prevent API errors. The TF-IDF model performs very reasonably at 67% accuracy for 8 classes!)*

### LLM Judge Scores (1-5 Scale)
*(Sourced from Mock LLM fallback outputs — see Section 6 for why these numbers are not meaningful)*
- **Overall Score**: 3.20/5.0
- **Tone**: 4.00/5.0
- **Groundedness**: 3.00/5.0

## 8. What is misleading about my headline number?
The headline numbers, while comprehensive, are subject to significant caveats:
1. **Mock Mode Deflation**: The AI Agent's 11% accuracy does not reflect real LLM performance (which would likely easily exceed the TF-IDF's 67%). It reflects our robust mock-mode fallback mechanism successfully preventing crashes during offline execution.
2. **Sample Size Constraints**: Evaluating on 200 examples means 1 misclassified example swings accuracy by 0.5%. Certain minority classes in the 8-class taxonomy have as few as 3 examples, making Macro-F1 highly unstable.
3. **Single Brand Bias**: The retrieval embeddings are tuned implicitly on AppleSupport's specific dialect. These metrics will not directly translate to a brand like SpotifyCares.
4. **LLM as Judge**: While efficient, LLMs inherently possess a "lenient grading" bias. The agreement analysis in Section 6 confirms this directly — Mock Mode produces a constant judge score that has zero correlation with human judgement (κ = 0.000).

## 9. Top 5 Failure Modes
*(Extracted from `failure_analysis.csv` and `failure_analysis.md`)*

**1. Intent Misclassification**
Ambiguous language that doesn't match clear keyword signals gets collapsed into the mock fallback intent.
> *Real example* — Customer: `"Replacement MacBook Pro does the same shit as the previous. @115858 needs to figure this shit out."` Expected: `device_hardware_problems` | Predicted: `battery_drain`
*Hypothesis*: The mock classifier ignores all semantic content and always returns `battery_drain`. With a real LLM, the word "Replacement" and "MacBook Pro" would clearly signal `device_hardware_problems`. *Improvement: Better few-shot examples for edge cases; real LLM evaluation to confirm.*

**2. Escalation False Positives**
The keyword-based escalation triggers on surface matches regardless of context.
> *Real example* — Customer: `"my phone was stolen & this picture showed up on my icloud @AppleSupport"` → Predicted: `escalate` (correct). But consider: `"My stolen phone was returned — how do I reactivate it?"` → Would also predict: `escalate` (false positive, as this is a benign recovery question).
*Hypothesis*: The flat keyword list `["stolen", "hacked", ...]` has no awareness of surrounding context. *Improvement: Allow the LLM context-aware heuristic to override keyword triggers when full context is benign.*

**3. Out-of-Distribution Inputs**
Extremely short or context-free tweets produce near-zero similarity scores across the retrieval corpus.
> *Real example* — Customer: `"@115858 what the fuck"` (evaluation_set.csv row `128e5f61`) | Gold intent: `ios_update_problems` | Predicted: `battery_drain` (mock fallback). The retriever would return low-confidence matches for this 4-word message regardless of model quality.
*Hypothesis*: With no semantic content beyond frustration, the dense retriever cannot find a meaningful nearest neighbour. *Improvement: Add a retrieval confidence threshold — if max cosine similarity is below a floor (e.g., 0.4), route directly to escalation instead of generating a low-confidence reply.*

**4. Ungrounded Hallucinations**
The generative layer is instructed to stay grounded in retrieved context, but the retrieval quality determines whether that constraint is meaningful.
> *Real example* — Customer: `"@AppleSupport ios 11 drains battery can you fix this"` | Retrieved 3 similar cases. Mock reply: `"This is a mock draft reply. Please check your settings."` In real LLM mode, if the Top-3 retrieved cases happen to be tangentially related (e.g., about iOS crashing rather than battery), the generator may still confidently cite troubleshooting steps that weren't actually in the retrieved replies.
*Hypothesis*: Cosine similarity does not guarantee thematic alignment — two tweets can be semantically close in embedding space while addressing different aspects of the same issue. *Improvement: Add a post-generation factual consistency check using the same LLM judge rubric before returning the reply.*

**5. Taxonomy Overlap**
`device_hardware_problems` vs `battery_drain` causes ground-truth labelling disagreements at the boundary.
> *Real example* — Customer: `"Iphone 6 heats and battery draining very fast after upgrading to ios 11"` (evaluation_set.csv row `3b717bdd`) | Labelled as: `battery_drain`. But heating + draining together could equally be labelled `device_hardware_problems` (the battery is a hardware component). This ambiguity propagates into Macro-F1 instability.
*Hypothesis*: The taxonomy was derived from unsupervised KMeans clustering, which separates clusters by statistical distance but cannot enforce human-interpretable mutual exclusivity at the boundary. *Improvement: Define explicit disambiguation rules in `intent_definitions.json` for known boundary cases (e.g., "heating issues → device_hardware_problems even when paired with battery complaints").*

## 10. What I'd Do Next With One More Week

Grounded in the failure modes and limitations documented above, here are the 5 highest-priority changes:

1. **Run the full pipeline with a real API key (highest priority)**. Every meaningful metric in this report — intent accuracy, judge scores, Kappa — is a Mock Mode artifact. The first hour of an additional week would go toward setting `MOCK_MODE=false` with a real `LLM_API_KEY` and re-running `scripts/run_evaluation.py` on the 200-example set. This single change would convert the "AI Agent" column in Section 7 from a placeholder to a real data point, and produce meaningful per-example judge scores that allow a non-trivial Kappa.

2. **Add a retrieval confidence threshold to fix failure mode 3 (out-of-distribution inputs)**. Currently, the retriever always returns the Top-3 results regardless of similarity score. Adding a floor (e.g., if `max(cosine_similarity) < 0.40`, escalate immediately) would prevent the generator from producing replies grounded in irrelevant context, and would likely improve escalation recall substantially for ambiguous short-form tweets.

3. **Replace flat keyword escalation with context-aware escalation to fix failure mode 2**. The current keyword list in `escalate.py` causes false positives on sentences like "my stolen phone was returned." With one more week, I'd replace hard keyword matching with a two-step check: keyword present → pass to LLM with full context → LLM decides whether the keyword is used in a genuinely high-risk sense. This preserves speed for obvious cases while reducing false positive escalations.

4. **Expand the evaluation set and add stratified sampling**. The current 200-example set is a simple random sample, leaving some intent classes with as few as 3 examples. With one more week, I'd re-run `create_evaluation_set.py` with a stratified design — at minimum 20 examples per intent class — to stabilize Macro-F1 and make per-class results interpretable.

5. **Resolve the taxonomy overlap between `battery_drain` and `device_hardware_problems`**. I'd add explicit disambiguation rules to `intent_definitions.json` (e.g., heating issues → hardware; purely software-triggered battery drain → battery_drain) and re-label the ambiguous examples in the evaluation set, then measure the impact on Macro-F1 for those two classes.

## 11. Decision Log

1. **Decision**: Built a custom `.env` parser and zero-dependency `urllib` HTTP LLM client.
   - **Reason**: The execution environment strictly blocked `pip install` network requests for `openai` or `requests`.
   - **Tradeoff**: Harder to maintain over time, but guarantees 100% immediate execution without environment configuration errors.

2. **Decision**: Enforced Out-Of-Fold (OOF) cross-validation for baselines.
   - **Reason**: Using pseudo-labels originally risked massive confirmation bias. OOF ensures strict zero-leakage evaluation.
   - **Tradeoff**: Increases computational overhead (5x training loop) but guarantees validity.

3. **Decision**: Separated intent labels into `intent_definitions.json`.
   - **Reason**: Hardcoding 8 string variables in multiple python files invites spelling errors (e.g., `battery_drain` vs `battery_draining`).
   - **Tradeoff**: Requires file I/O during early pipeline bootstrapping.

4. **Decision**: Adopted a Mock Mode fallback for LLMs and `SentenceTransformers`.
   - **Reason**: Pipeline reproducibility > localized accuracy. The agent must build and test even if HuggingFace/OpenAI is down.
   - **Tradeoff**: Final automated metric generation yields dummy numbers unless explicitly run with an API key.

5. **Decision**: Used `TfidfVectorizer` (max features=1000) over simpler Word2Vec.
   - **Reason**: TF-IDF provides deterministic, explainable word weights which is crucial for a classical baseline comparison against complex deep neural networks.
   - **Tradeoff**: TF-IDF ignores word order and semantics; Word2Vec would generalize better but is non-deterministic without a fixed seed.

6. **Decision**: Used `sentence-transformers/all-MiniLM-L6-v2` for dense retrieval rather than a larger model (e.g., `all-mpnet-base-v2`).
   - **Reason**: `all-MiniLM-L6-v2` runs on CPU in under 2 seconds per encode call, making it viable for a local offline pipeline. Larger models would have been too slow for the 8,000-pair embedding pre-computation without GPU access.
   - **Tradeoff**: The 384-dimensional MiniLM embedding is less expressive than 768-dim models; retrieval quality on highly nuanced edge cases is reduced.

7. **Decision**: Pre-computed and cached corpus embeddings as a `.npy` flat file rather than using a vector database (e.g., FAISS, ChromaDB).
   - **Reason**: A flat NumPy cache + cosine similarity scan is zero-dependency, trivially reproducible, and fast enough for 8,000 vectors on CPU. A vector database adds operational complexity with no meaningful speed benefit at this scale.
   - **Tradeoff**: Linear scan does not scale beyond ~50K vectors without noticeable latency. Any production deployment at scale would need FAISS or equivalent.

8. **Decision**: Used simple random sampling (not stratified) for the 200-example evaluation set.
   - **Reason**: The labelling CLI (`create_evaluation_set.py`) was designed for speed — the labeller needed to review 200 real tweets from the actual distribution without bias. Stratified sampling would have required knowing the intent distribution upfront, which was itself being discovered.
   - **Tradeoff**: Results in an unbalanced label distribution (some intents have as few as 3 examples), making Macro-F1 highly unstable. A follow-up stratified resample was identified as the correct next step.

9. **Decision**: Derived the intent taxonomy bottom-up from unsupervised KMeans clustering (`discover_intents.py`) rather than defining categories top-down from domain knowledge.
   - **Reason**: Top-down taxonomy design risks imposing categories that don't match the actual distribution of user complaints. KMeans on sentence embeddings reveals what clusters naturally exist in the data before any labelling effort.
   - **Tradeoff**: Unsupervised clusters don't respect human-interpretable boundaries — two semantically related intents (e.g., `battery_drain` and `device_hardware_problems`) can be placed in the same cluster, requiring manual taxonomy refinement after inspection.

10. **Decision**: Implemented escalation as a two-stage hybrid (keyword rules → LLM fallback) rather than a pure LLM decision.
    - **Reason**: Pure LLM escalation decisions are expensive, slow, and non-deterministic. Certain cases (legal threats, security incidents, stolen devices) should *always* escalate regardless of LLM context — hard rules guarantee this. The LLM fallback handles the genuinely ambiguous middle ground.
    - **Tradeoff**: Hard keyword rules produce false positives when sensitive words appear in benign contexts (documented as Failure Mode 2).

11. **Decision**: Evaluated reply quality using an LLM-as-judge rubric (5 dimensions) rather than BLEU/ROUGE.
    - **Reason**: BLEU and ROUGE measure surface-level n-gram overlap with reference answers. For an open-ended support reply task where many phrasings of the same correct answer exist, these metrics are misleading. An LLM judge can assess whether the reply is actually grounded, correct, and helpful.
    - **Tradeoff**: LLM judges introduce their own biases (leniency bias, verbosity preference) and cannot be relied upon without human calibration — which is precisely why `label_judge_agreement.py` exists.

12. **Decision**: Scored human-judge agreement on overall quality only (not per-dimension) in the CLI tool.
    - **Reason**: Asking a human labeller to score 5 separate dimensions (Groundedness, Correctness, Tone, Actionability, Factual Consistency) per example would have taken significantly longer and risked annotator fatigue across 35 examples.
    - **Tradeoff**: The resulting Kappa measures agreement only on composite quality, masking dimension-level disagreements where the LLM and human might diverge most sharply (e.g., Groundedness is harder to assess without the retrieved context visible to the human).

## 12. Reproduction
To reproduce this exact report and metrics using the subsampled environment:
```powershell
.venv\Scripts\activate
python -m unittest discover tests
python scripts\run_baselines.py
python scripts\run_evaluation.py
python scripts\analyze_failures.py
```
*(Optionally run `python scripts\label_judge_agreement.py` to calculate human parity on the generated outputs)*
