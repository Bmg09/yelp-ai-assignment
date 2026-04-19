# Yelp AI Assignment — Prompt Engineering & Fine-Tuning

Python + Jupyter · Vercel AI Gateway · DistilBERT on M4 MPS

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name yelp-ai --display-name "Yelp AI"

echo "AI_GATEWAY_API_KEY=vck_..." > .env

jupyter notebook notebooks/00_setup.ipynb
```

Then run notebooks in order: 01 → 02 → 03 → 04a → 04b → 05.

## What each notebook does

| Notebook | Task | Core question |
|---|---|---|
| `00_setup.ipynb` | Env check | Gateway + 3 classifiers + judge + datasets intact |
| `01_json_prompt.ipynb` | Task 1 | Zero vs few-shot across 3 classifier families — pick winner |
| `02_cot_vs_direct.ipynb` | Task 2 | Does CoT help 5-way ordinal classification? |
| `03_multi_objective.ipynb` | Task 3 | Stars + signal + response, judged by different-family model |
| `04a_domain_shift.ipynb` | Task 4a | Yelp → Amazon/IMDB accuracy drop |
| `04b_distilbert.ipynb` | Task 4b | DistilBERT fine-tune on M4 MPS, cross-domain eval |
| `05_adversarial.ipynb` | Robustness | 7 attacks + self-consistency mitigation |

## Models

| Role | Model | Rationale |
|---|---|---|
| Classifier (active) | `deepseek/deepseek-v3.2` | Won bake-off in notebook 01 |
| Classifier candidates | `openai/gpt-5-nano`, `google/gemini-2.0-flash` | Compared in notebook 01 |
| Judge | `anthropic/claude-haiku-4.5` | Different family, avoids self-judge bias |
| Encoder FT | `distilbert-base-uncased` (66M params) | M4 MPS, ~15 min for 10k × 3e |

## Layout

```
lib/
  config.py          # MODELS registry, env loading
  schemas.py         # Pydantic classes
  gateway.py         # AsyncOpenAI wrapper, provider-aware parsing, dual cache
  cache.py           # diskcache SHA256
  prompts.py         # system strings + builders
  datasets.py        # JSONL loader + class_dist
  metrics.py         # sklearn-backed
  concurrency.py     # asyncio.Semaphore + tqdm
  adversarial.py     # 7 attack fns
  plots.py           # confusion heatmap, bar compare

notebooks/           # 7 Jupyter notebooks
scripts/             # distilbert_{train,eval}.py
data/                # JSONL eval sets (git-ignored)
results/             # per-notebook JSON + plots/
report/report.md     # final writeup
archive/ts/          # deprecated TS implementation (reference only)
```

## Caching

- **Local disk** (`.pycache/`): SHA256 of `(model, system, prompt, schema_json, temperature)`. Temp=0 only. Disable via `AI_CACHE=0`.
- **Gateway auto** (Vercel): `providerOptions.gateway.caching: 'auto'` — OpenAI prefix cache, saves ~50% on shared few-shot tokens.

## Cost

All experiments end-to-end: <$1 API + $0 local FT.

## Key findings (condensed)

- Few-shot beats zero by 4-6pp accuracy across models.
- DeepSeek V3.2 > Gemini 2.0 Flash > gpt-5-nano on 5-way ordinal Yelp.
- CoT hurts accuracy and triples latency on this task.
- Multi-objective prompt *improves* classification (+4.7pp).
- Sarcasm is hardest attack; self-consistency doesn't help (surface bias is shared).
- IMDB cross-domain drop is partly binary-vs-ordinal metric artifact.

See `report/report.md` for full writeup.
