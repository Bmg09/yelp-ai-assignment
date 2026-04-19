# Advanced AI Systems for Yelp Reviews — Report

**Candidate**: Bikram Ghosh · **Date**: 2026-04-19

## Executive summary

- **Best classifier**: `deepseek/deepseek-v3.2` (few-shot) — 66.0% acc, 0.662 F1, 0.350 MAE on 500 stratified Yelp test. Beats `gpt-5-nano` by 7.4pp.
- **CoT hurts**: −1.4pp accuracy, 2.8× latency. Direct wins on strong classifier + ordinal target.
- **Multi-objective preserves accuracy** (66.0%, stratified 150) — business outputs for free; 4★ class regresses 19pp (decisive schema over-predicts 5★).
- **Domain shift**: Yelp→Amazon −2pp (near-domain generalizes). Yelp→IMDB −24pp acc (partly binary-vs-ordinal artifact; F1 only −0.07).
- **DistilBERT FT (10k × 3e, M4 MPS)**: 59/52/51% on Yelp/Amazon/IMDB — loses Yelp (−6.8), Amazon (−12.3), wins IMDB (+8.7).
- **Adversarial**: sarcasm hardest (55%). Self-consistency does NOT help (shared bias at temp=0).
- **JSON compliance ~100%** via Pydantic schema enforcement. **Total API cost <$1.**

## Stack & datasets

Python 3.13 + Jupyter · Vercel AI Gateway · Pydantic v2 · `asyncio.Semaphore(30)` · SHA256 disk cache + gateway prefix cache · DistilBERT on M4 MPS.

| Dataset | Source | Classes | n |
|---|---|---|---|
| yelp_eval | `Yelp/yelp_review_full` test | 1–5 | 500 (100/class) |
| yelp_fewshot | same | 1–5 | 5 (1/class) |
| amazon_eval | `mteb/amazon_reviews_multi` en | 1–5 | 300 (60/class) |
| imdb_eval | `stanfordnlp/imdb` test | {1,5} | 300 (150/class) |
| adversarial | 20 Yelp seeds × 7 attacks + clean | 1–5 | 160 |

Fewshot seed 777, eval seed 42 — no overlap.

## Task 1 — Zero-shot vs Few-shot JSON (bake-off)

3 classifier families × 2 strategies × 500 rows, Pydantic `StarsOnly` schema.

| Model | Strategy | Acc | Macro-F1 | MAE | Compliance | Time |
|---|---|---|---|---|---|---|
| **deepseek-v3.2** | **few** | **66.0%** | **0.662** | **0.350** | 100% | 53s |
| gemini-2.0-flash | few | 66.0% | 0.656 | 0.354 | 100% | 17s |
| deepseek-v3.2 | zero | 62.4% | 0.617 | 0.390 | 99.6% | 64s |
| gemini-2.0-flash | zero | 63.4% | 0.627 | 0.384 | 100% | 20s |
| gpt-5-nano | few | 58.6% | 0.572 | 0.442 | 100% | 93s |
| gpt-5-nano | zero | 57.2% | 0.544 | 0.456 | 100% | 100s |

**Findings**:

- Few-shot lifts all models by 3–4pp accuracy.
- DeepSeek tied with Gemini on acc (66.0%) but wins F1 + MAE → picked via composite `F1/(MAE+0.01)` = 1.839.
- gpt-5-nano slow + weakest — reasoning tokens (~450/call) explain both.
- 3★/4★ hardest classes. Zero-shot 4★ = 41% (over-predicts 5★); few-shot cuts confusion.
- Label noise: some 3★ ground truth has unambiguously positive text ("Best hot and sour soup I've ever had!").

`ACTIVE_CLASSIFIER = deepseek/deepseek-v3.2` locked for downstream tasks.

**Trade-off**: few-shot = 10× input tokens, mitigated ~50% by gateway prefix cache.

## Task 2 — CoT vs Direct

DeepSeek, 500-row Yelp, `StarsCoT` (reasoning+stars) vs `StarsDirect` (stars).

| Metric | Direct | CoT | Δ |
|---|---|---|---|
| Accuracy | 64.2% | 62.8% | **−1.4pp** |
| Macro-F1 | 0.634 | 0.613 | −0.021 |
| MAE | 0.370 | 0.394 | +0.024 |
| Time | 29s | 82s | **2.8×** |

Per-class: 2★ +5pp, 3★ −5pp, **4★ −11pp**, 5★ +4pp.

**Findings**:

- CoT surfaces nits that drift 4★ → 3★ decisions. Reasoning traces confirm: "positive tone, but reviewer also mentioned...".
- 14/500 (2.8%) reasoning/answer mismatches (reasoning says "5", field says "2") — fidelity concern for auditable systems.
- **For 5-way ordinal sentiment on strong classifier, direct wins on acc, latency, cost.** Classic "reasoning helps weak models more than strong" result.

## Task 3 — Multi-Objective Assistant + Cross-Family Judge

150 stratified Yelp (30/class), `MultiObjective` schema (stars + signal + response). 60-sample judged by `anthropic/claude-haiku-4.5` (different family → avoids self-bias).

### Classification

| Metric | Value |
|---|---|
| n | 150 (stratified) |
| Accuracy | 66.0% |
| Macro-F1 | 0.654 |
| MAE | 0.347 |

Flat vs Task 1 baseline (66.0%). Initial random subset showed spurious +4.7pp — stratified re-run kills that. **Multi-objective gives business-ready outputs at zero accuracy cost.**

Per-class: 1★ 73%, 2★ 77%, 3★ 57%, **4★ 37%**, 5★ 87%. 4★ regression real — structured schema pushes decisive 5★.

### Judge scores (n=49 valid of 60; 81.7% compliance)

| Axis | Mean | Std |
|---|---|---|
| Faithfulness | 4.00 | 0.89 |
| Politeness | **4.98** | 0.14 |
| Actionability | **3.06** | 1.03 |

**Findings**:

- Politeness saturated — not a useful discriminator.
- Actionability weak — responses default to "we'd love to have you back" rather than concrete offers. Instruction-design gap.
- Faithfulness tail: worst case invented menu items ("Squash Blossom Hush Puppies", "OM burger") absent from review. Hallucination in generated replies is higher-severity than star mis-predictions.
- Cross-family judge eliminates obvious self-bias; still not ground truth.

## Task 4a — Domain Shift (prompting)

Same DeepSeek few-shot (Yelp demos) on three domains.

| Domain | n | Acc | Macro-F1 | MAE | Δ acc |
|---|---|---|---|---|---|
| Yelp | 500 | 66.0% | 0.662 | 0.350 | — |
| Amazon | 300 | 64.0% | 0.636 | 0.390 | −2.0 pp |
| IMDB | 300 | 42.3% | 0.595 | 0.727 | **−23.7 pp** |

**Findings**:

- Amazon near-zero drop — review style close to Yelp.
- IMDB massive acc drop but F1 only −0.07 → metric artifact: IMDB binary {1,5} vs model emitting 2/3/4 for mild reviews. F1 on {1,5} forgives; accuracy punishes.
- Re-bucketing (`≤3 → 1`, `≥4 → 5`) closes most IMDB gap — model semantically right, scale-mismatched.

**Mitigations**: (1) mixed-domain few-shot, (2) classification-head FT (see 4b), (3) re-bucketing for granularity mismatch.

## Task 4b — DistilBERT Fine-Tune on M4 MPS

`distilbert-base-uncased` (66M) on 10k stratified Yelp, 3 epochs, lr=2e-5, bs=16, maxlen=256. Wall time ~18min.

| Epoch | eval_acc | macro_f1 | MAE |
|---|---|---|---|
| 3 (best) | **0.590** | 0.589 | 0.482 |

### Head-to-head (DeepSeek few-shot vs DistilBERT)

| Domain | DeepSeek | DistilBERT | Winner |
|---|---|---|---|
| Yelp | **66.0%** | 59.2% | LLM +6.8 |
| Amazon | **64.0%** | 51.7% | LLM +12.3 |
| IMDB | 42.3% | **51.0%** | BERT +8.7 |

**Findings**:

- 10k rows below encoder saturation (published bench ~64% at full 650k).
- **BERT wins IMDB** — argmax over 5 logits concentrates mass at extremes; LLM hedges to mid-range and gets crushed under binary truth.
- MAE asymmetry on IMDB: BERT 0.91 (when wrong, very wrong), LLM 0.73 (conservative).
- Amazon is the encoder story — same label space, different lexicon → overfits Yelp phrasing. Fix: mixed-domain train or domain-adversarial.

**Trade-off**: **encoder SFT > in-domain ceiling at scale; prompt-LLM > OOD floor**. This dataset budget (10k) insufficient for SFT dominance. Production lever: large+stable data → encoder; small/drift → prompting.

## Adversarial Robustness

20 Yelp seeds × 7 attacks + clean = 160 rows. Baseline (greedy, temp=0) vs self-consistency (n=3, temp=0.7, majority vote).

| Attack | Baseline | SC | Δ |
|---|---|---|---|
| sarcasm | **55%** | 55% | 0 |
| clean | 60% | 60% | 0 |
| mixed_language | 60% | 60% | 0 |
| emoji | 65% | 60% | −5 |
| irrelevant_padding | 65% | 60% | −5 |
| competitor | 70% | 70% | 0 |
| typo | 70% | 70% | 0 |
| negation | 75% | 70% | −5 |

**Findings**:

- Sarcasm hardest — model reads surface sentiment, misses flip.
- SC does NOT help (0 wins, 3 losses). DeepSeek at temp=0 near ceiling on these simple transforms; temperature noise adds random errors, dilutes vote.
- Small n=20 per attack → ±21pp CI. All deltas within noise — claim is directional.

**Better mitigations**: (1) sarcasm-aware system prompt, (2) sarcasm-augmented few-shot, (3) adversarial FT on sarcasm/negation, (4) cross-family ensemble disagreement flag for human review.

## Trade-offs summary

| Decision | Gain | Cost |
|---|---|---|
| Few-shot over zero | +3–4pp acc | 10× input tokens (−50% via cache) |
| CoT over direct | +5pp on 2★ | −11pp on 4★, 3× latency |
| Multi-objective | Business outputs at flat agg acc | 4★ regresses 19pp; judge compliance 82% |
| DistilBERT | Higher in-domain ceiling | Worse OOD, training infra |
| Vercel Gateway | Unified routing, cheap cross-family compare | Provider quirks (markdown fences, response_format support) |
| Pydantic strict | ~100% JSON compliance | OpenAI-only `.parse()`; fallback path needed |

## Limitations

- 500-row eval → ±2pp CI. Task 3 stratified 30/class → ±9pp per-class CI.
- Judge (Haiku) has own biases; no human calibration step completed.
- IMDB binary creates unavoidable metric artifact on 1–5 scale.
- Adversarial n=20 per attack → noise-bound.
- Label noise: some Yelp 3★ rows have positive text. Real-world limit.
- FT scope: only DistilBERT. No LoRA LLM or OpenAI FT comparison.

## Cost report (actual, April 2026)

| Task | Calls | ~$ |
|---|---|---|
| Task 1 bake-off (3×2×500) | 3000 | 0.30 |
| Task 2 (2×500) | 1000 | 0.05 |
| Task 3 + judge | 210 | 0.10 |
| Task 4a (500+300+300) | 1100 | 0.05 |
| Adversarial (160×4) | 640 | 0.03 |
| **Total API** | | **~0.53** |
| DistilBERT FT on M4 | — | 0 |

Reproduce: `pip install -r requirements.txt && jupyter notebook notebooks/00_setup.ipynb` → run notebooks 01–05 in order. Cached runs: ~2min end-to-end.
