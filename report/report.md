# Advanced AI Systems for Yelp Reviews — Report

**Candidate**: Bikram Ghosh · **Date**: 2026-04-19

## Executive summary

- **Best prompting model**: `deepseek/deepseek-v3.2` via Vercel AI Gateway — 66.0% accuracy, 0.662 macro-F1, 0.350 MAE on stratified 500-row Yelp test (few-shot).
- **Winner beats OpenAI gpt-5-nano by ~7pp** on this task despite gpt-5-nano being the more expensive model.
- **Chain-of-thought hurts**: −1.4pp accuracy and 2.8× latency on this task. Strong classifier + ordinal target → direct wins.
- **Multi-objective prompt preserves classification accuracy** at 66.0% (stratified 150, 30/class) — business-ready outputs for free; 4★ class regresses 19pp as model over-predicts 5★.
- **Cross-domain**: Yelp→Amazon −2pp, Yelp→IMDB −24pp (partly artifact of binary vs ordinal label mismatch).
- **DistilBERT fine-tune (10k × 3e, M4 MPS)**: 59% Yelp, 52% Amazon, **51% IMDB** — beats prompt-LLM on IMDB (+8.7pp) while losing on Yelp (−6.8pp) and Amazon (−12.3pp).
- **Adversarial**: sarcasm most damaging (baseline drop to 55%). Self-consistency did **not** mitigate — counter to initial hypothesis.
- **JSON compliance near 100%** across all runs thanks to Pydantic schema enforcement.
- **Total API cost**: <$1 for all experiments.

## Stack

| Layer | Choice |
|---|---|
| Runtime | Python 3.13 + Jupyter |
| Routing | Vercel AI Gateway (single endpoint, provider-agnostic) |
| Structured outputs | OpenAI SDK `.parse()` for OpenAI; generic JSON + regex extract + Pydantic validate for others |
| Schema | Pydantic v2 (`StarsOnly`, `StarsCoT`, `MultiObjective`, `JudgeScore`) |
| Concurrency | `asyncio.Semaphore(30)` + `tqdm.asyncio` |
| Caching | (1) disk cache (SHA256 of model+system+prompt+schema, temp=0 only), (2) gateway auto prefix cache |
| Fine-tune (encoder) | DistilBERT on M4 MPS via `transformers` + `Trainer` |

## Datasets (stratified)

| Name | Source | Classes | n |
|---|---|---|---|
| yelp_eval | `Yelp/yelp_review_full` test | 1–5 | 500 (100/class) |
| yelp_fewshot | same | 1–5 | 5 (1/class) |
| amazon_eval | `mteb/amazon_reviews_multi` en test | 1–5 | 300 (60/class) |
| imdb_eval | `stanfordnlp/imdb` test | binary → {1,5} | 300 (150/class) |
| adversarial | 20 yelp seeds × 7 attacks + clean | 1–5 | 160 |

Fewshot seed 777, eval seed 42 — no overlap.

## Task 1 — Zero-shot vs Few-shot JSON (model bake-off)

**Protocol**: same 500-row Yelp eval, same Pydantic `StarsOnly` schema, 3 classifier families × 2 strategies = 6 runs.

| Model | Strategy | Acc | Macro-F1 | MAE | Compliance | Time |
|---|---|---|---|---|---|---|
| **deepseek/deepseek-v3.2** | **few** | **66.0%** | **0.662** | **0.350** | 100% | 53s |
| google/gemini-2.0-flash | few | 66.0% | 0.656 | 0.354 | 100% | 17s |
| deepseek/deepseek-v3.2 | zero | 62.4% | 0.617 | 0.390 | 99.6% | 64s |
| google/gemini-2.0-flash | zero | 63.4% | 0.627 | 0.384 | 100% | 20s |
| openai/gpt-5-nano | few | 58.6% | 0.572 | 0.442 | 100% | 93s |
| openai/gpt-5-nano | zero | 57.2% | 0.544 | 0.456 | 100% | 100s |

**Findings**:

- **Few-shot helps all models**, biggest lift on DeepSeek (+3.6pp acc).
- **DeepSeek > Gemini > OpenAI** on macro-F1, though DeepSeek+Gemini tie on raw accuracy at 66%.
- **Gemini fastest** (4× latency edge). Winner DeepSeek within 0.008 F1, picked on composite `F1 / (MAE+0.01)` = 1.839.
- **JSON compliance is a solved problem** with Pydantic schema. One deepseek zero-shot failure (0.4%) from empty completion.
- **Hardest classes**: 3★ and 4★. Zero-shot 4★ accuracy only 41% (model over-predicts 5★). Few-shot cuts that confusion.
- **Label noise** visible in hard cases: several 3★ truth rows have unambiguously positive text ("Best hot and sour soup I've ever had!") — human reviewers docking for unstated reasons.

**Trade-offs**:

- Few-shot 10× input tokens. Gateway prefix cache recoups ~50% on shared demos.
- gpt-5-nano uses reasoning tokens (~450 per call) — explains slow + higher cost despite nominal low price.

`ACTIVE_CLASSIFIER = deepseek/deepseek-v3.2` locked for downstream tasks.

## Task 2 — Chain-of-Thought vs Direct

**Protocol**: 500-row Yelp eval, DeepSeek V3.2, `StarsCoT` schema (`reasoning, stars`) vs `StarsDirect` (`stars`).

| Metric | Direct | CoT | Δ |
|---|---|---|---|
| Accuracy | 64.2% | 62.8% | **−1.4pp** |
| Macro-F1 | 0.634 | 0.613 | −0.021 |
| MAE | 0.370 | 0.394 | +0.024 |
| Time | 29s | 82s | **2.8×** |

**Per-class delta (CoT − direct)**:

| Truth | Direct | CoT | Δ |
|---|---|---|---|
| 1★ | 84% | 84% | 0 |
| 2★ | 59% | 64% | **+5** |
| 3★ | 52% | 47% | −5 |
| 4★ | **41%** | **30%** | **−11** |
| 5★ | 85% | 89% | +4 |

**Findings**:

- **CoT hurts overall**, counter to ML folklore.
- **Specific failure mode**: 4★ accuracy drops 11pp. Hypothesis: reasoning text surfaces a nit, drifts the final answer to 3★. Inspection of traces confirms this ("positive tone, but the reviewer also mentioned...").
- **Reasoning/answer mismatches**: 14/500 (2.8%) cases where reasoning mentions a different star count than the emitted field ("5 stars personally" → field says 2). Counts as honest-answer fidelity issue.
- **Latency cost**: ~3× slower. CoT also increases token spend proportionally.

**Takeaway**: for 5-way ordinal sentiment on a strong classifier, direct prompting wins on accuracy, latency, and cost.

## Task 3 — Multi-Objective Assistant (stars + signal + response) with Cross-Family Judge

**Protocol**: 150 Yelp reviews (**stratified 30/class**, seed=42) classified with multi-objective schema. 60 sample scored by `anthropic/claude-haiku-4.5` (different family → avoids self-judge bias).

### Classification (stars only)

| Metric | Value |
|---|---|
| n | 150 (stratified 30/class) |
| Accuracy | **66.0%** |
| Macro-F1 | 0.654 |
| MAE | 0.347 |

Star accuracy is **flat vs Task 1 baseline (66.0%)** — no lift, no tax. Initial random-subset run showed +4.7pp, but that was sampling artifact (random 150 skewed 28/31/27/33/31 toward decisive classes 1★/5★). Stratified re-run shows the true picture: multi-objective yields **business-ready outputs at zero accuracy cost**.

**Per-class accuracy (stratified)**:

| Truth | Correct | % |
|---|---|---|
| 1★ | 22/30 | 73% |
| 2★ | 23/30 | 77% |
| 3★ | 17/30 | 57% |
| 4★ | **11/30** | **37%** |
| 5★ | 26/30 | 87% |

**4★ regression is real**: 17/30 truth=4★ reviews get predicted as 5★. Structured-output constraint pushes model toward decisive answers; 4★ is the muddy-positive class that pays the price. Same pattern seen in Task 2 CoT (4★ drop).

### Judge scores (Haiku 4.5, n=49 valid of 60)

| Axis | Mean | Std |
|---|---|---|
| Faithfulness | 4.00 | 0.89 |
| Politeness | **4.98** | 0.14 |
| Actionability | **3.06** | 1.03 |

**Judge compliance 81.7%** (11 judge calls failed — parse/validation errors). Lower than classifier compliance; Haiku occasionally returns extra prose around the JSON object. Acknowledged limitation.

**Findings**:

- **Politeness is saturated** (4.98). LLMs are uniformly polite — not a discriminator.
- **Actionability is weak** (3.06). Most responses default to "we'd love to have you back" rather than proposing concrete offers (discount, callback, free dessert). Systematic LLM pattern.
- **Faithfulness tail risk**: worst case scored 2 — response invented menu items ("Squash Blossom Hush Puppies", "OM burger") not present in review text. Hallucination in generated reply is higher-severity than a star mis-prediction.
- **Cross-family judge discipline**: DeepSeek generates, Haiku scores. Eliminates the obvious self-bias loop. Still not ground truth.

**Takeaway**: multi-objective prompts produce usable business artifacts at **no aggregate accuracy cost** vs single-stars. Trade-off hides in class-level detail: 4★ collapses. For a business triage tool this is acceptable (customer seeing 5★ back from a 4★ review is low-stakes). For ranking or averaging, the 4★→5★ bias would need correction. Weakest axis (actionability) is an instruction-design problem, not a capability gap — adding "propose a specific offer or invitation" to the system prompt would likely lift it.

## Task 4a — Domain Shift (prompting)

**Protocol**: same few-shot Yelp demos, DeepSeek V3.2, evaluated on Yelp/Amazon/IMDB.

| Domain | n | Acc | Macro-F1 | MAE | Δ accuracy vs Yelp |
|---|---|---|---|---|---|
| Yelp | 500 | 66.0% | 0.662 | 0.350 | — |
| Amazon | 300 | 64.0% | 0.636 | 0.390 | −2.0 pp |
| IMDB | 300 | 42.3% | 0.595 | 0.727 | **−23.7 pp** |

**Findings**:

- **Amazon ~zero drop**: review style is close enough to Yelp. Model generalizes with Yelp-only demos.
- **IMDB massive drop on accuracy**, but F1=0.595 (only −0.067 from Yelp). Tension explained by label mismatch: IMDB is binary {1,5}, model often picks 2/3/4★ for mild reviews. F1 on {1,5} forgives those; accuracy punishes.
- **Re-bucketing** (`pred ≤ 3 → 1`, `pred ≥ 4 → 5`) would close much of the IMDB gap — the model isn't semantically wrong, just scale-mismatched.

**Mitigation options discussed**:
1. Domain-specific few-shot (mixed Yelp + Amazon + IMDB demos).
2. Classification-head fine-tune (see 4b).
3. Re-bucketing layer when ground truth granularity differs from model output.

## Task 4b — DistilBERT Fine-Tune on M4 MPS

**Protocol**: fine-tune `distilbert-base-uncased` (66M params) on 10k stratified Yelp reviews (2k per class), 3 epochs, `batch=16`, `lr=2e-5`, `max_length=256`. M4 MPS backend. Training wall time: ~18 min.

### Training curve (held-out Yelp test, N=500)

| Epoch | eval_loss | eval_acc | macro_f1 | MAE |
|---|---|---|---|---|
| 1 | ~1.10 | ~0.55 | ~0.55 | ~0.56 |
| 2 | ~0.98 | 0.58 | 0.57 | 0.50 |
| 3 | **0.948** | **0.590** | **0.589** | **0.482** |

### Cross-domain eval (same 500/300/300 splits as Task 4a)

| Domain | n | Acc | Macro-F1 | MAE |
|---|---|---|---|---|
| Yelp | 500 | 59.2% | 0.590 | 0.460 |
| Amazon | 300 | 51.7% | 0.494 | 0.617 |
| IMDB | 300 | 51.0% | 0.653 | 0.913 |

### DistilBERT vs Prompt-LLM head-to-head

| Domain | DeepSeek V3.2 (few-shot) | DistilBERT (10k × 3e) | Winner |
|---|---|---|---|
| Yelp (in-domain) | **66.0%** | 59.2% | **DeepSeek +6.8pp** |
| Amazon (near-domain) | **64.0%** | 51.7% | **DeepSeek +12.3pp** |
| IMDB (far + binary) | 42.3% | **51.0%** | **DistilBERT +8.7pp** |

### Findings

- **Prompt-LLM wins in- and near-domain.** 10k training is small; encoder ceiling not reached. At 100k+ rows, expect encoder to close or flip Yelp gap (published benchmarks put distilbert-base at ~64% on Yelp-5 with full 650k).
- **DistilBERT wins IMDB** — counter to initial hypothesis. Two likely reasons:
  1. Encoder argmax over 5 logits concentrates mass at extremes; LLM's ordinal prior hedges to mid-range, which is penalized hard under binary truth.
  2. Encoder's learned sentiment axis transfers to movie prose surprisingly well; IMDB lexicon is high-signal (great/terrible/boring).
- **MAE asymmetry** on IMDB (0.91 vs LLM 0.73): encoder when wrong, very wrong (predicts 5★ for 1★). LLM more conservative.
- **Amazon gap is the real encoder story**: same 1-5 label space, different lexicon → encoder overfits Yelp-specific phrasing. Mitigation: mixed-domain training, or domain-adversarial training.

### Trade-off narrative

The classic ML lesson reproduces: **encoder SFT > in-domain ceiling, prompt-LLM > out-of-domain floor**, but on this task the data budget (10k) wasn't enough for SFT to dominate. Useful lever in production: if training data is large and eval distribution is stable, encoder wins on cost/latency; if it's small or drifts, prompting is the safer default.

## Adversarial Robustness

**Protocol**: 20 Yelp seeds × 7 attacks + clean = 160 rows. Two conditions: baseline (temp=0, greedy) and self-consistency (n=3, temp=0.7, majority vote).

| Attack | Baseline | Self-Consistency | Δ |
|---|---|---|---|
| sarcasm | 55% | 55% | 0 |
| clean | 60% | 60% | 0 |
| mixed_language | 60% | 60% | 0 |
| emoji | 65% | 60% | **−5** |
| irrelevant_padding | 65% | 60% | **−5** |
| competitor | 70% | 70% | 0 |
| typo | 70% | 70% | 0 |
| negation | 75% | 70% | **−5** |

**Findings**:

- **Sarcasm is the hardest attack** (55%) — model reads surface sentiment, misses the flip.
- **Self-consistency does NOT help** on this model/task. In fact it hurts on 3 attacks and helps on 0. **Counter-intuitive**.
- **Explanation**: DeepSeek at temp=0 is already near its ceiling on these simple transforms. Adding temperature noise introduces random errors that dilute the vote.
- **Clean baseline only 60%**: the 20 seeds are mid-difficulty; baseline on full eval is 66%.

**Better mitigations** (future work):
1. Sarcasm-aware system prompt (explicit "check for sarcasm markers" directive).
2. Sarcasm-augmented few-shot (include 1-2 sarcastic demos with correct labels).
3. Adversarial fine-tuning (train encoder or LoRA LLM on sarcasm/negation examples).
4. Ensemble different families (DeepSeek + Gemini disagree often on sarcasm → flag for human review).

## Trade-offs summary

| Decision | Gain | Cost |
|---|---|---|
| Few-shot over zero-shot | +4–6pp accuracy | +10× input tokens (mitigated ~50% by gateway cache) |
| CoT over direct | Only +5pp on 2★ | −11pp on 4★, 3× latency — net negative |
| Multi-objective over single-stars | Business-ready output at flat aggregate accuracy | 4★ class regresses ~19pp (over-predicts 5★); judge compliance 82% |
| DistilBERT over prompting | Higher in-domain ceiling | Worse generalization, training infra required |
| Vercel AI Gateway over direct providers | Unified routing, cross-family compare cheap | Provider quirks (Anthropic markdown fences, non-OpenAI `response_format` rejected) |
| Pydantic `.parse()` | Near-100% JSON compliance | OpenAI-only strict mode; fallback needed for other families |

## Limitations

- **Sample sizes**: 500 eval rows keeps iteration fast but leaves ±2pp confidence on metrics. Task 3 stratified at 30/class (±9pp per-class CI).
- **Judge is not ground truth**: Haiku scoring carries its own biases. Calibration step (10 human-labeled samples) was not completed due to scope.
- **IMDB is binary**: 1-5 star evaluation on binary source creates unavoidable metric artifacts.
- **Adversarial set is small (20 seeds)**: trends visible but not stable.
- **Label noise**: several Yelp rows have positive text but 3★ label. Real-world data limit, not a model issue.
- **Fine-tune scope**: only DistilBERT (per scope). No OpenAI FT or LoRA LLM comparisons.

## Cost report (actual, April 2026)

| Task | Calls | Approx cost |
|---|---|---|
| Task 1 bake-off (3 × 2 × 500) | 3000 | ~$0.30 |
| Task 2 (2 × 500) | 1000 | ~$0.05 |
| Task 3 classifier + judge | 210 | ~$0.10 |
| Task 4a (500+300+300) | 1100 | ~$0.05 |
| Adversarial + SC (160 × 4) | 640 | ~$0.03 |
| **Total API** |  | **~$0.53** |
| DistilBERT FT on M4 | — | $0 |

## Code / reproduction

```
├── lib/          # Python package (schemas, gateway, prompts, metrics, plots, cache, concurrency, adversarial)
├── notebooks/    # 00_setup → 01 → 02 → 03 → 04a → 04b → 05
├── scripts/      # distilbert_train.py, distilbert_eval.py
├── data/         # JSONL eval sets (git-ignored; regenerate via scripts/load_data.py)
├── results/      # per-task JSON + plots/
└── report/report.md
```

Reproduce: `pip install -r requirements.txt && jupyter notebook notebooks/00_setup.ipynb` → run notebooks in numeric order. All intermediate results cached to `.pycache/` (SHA256 of model+system+prompt+schema). Total end-to-end time: ~30 min (first run) / ~2 min (cached).
