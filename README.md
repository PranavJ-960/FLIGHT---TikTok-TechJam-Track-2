# KuaiRand-Pure Autonomous ML Research Agent

An end-to-end pipeline for TechJam Track 2: an agent that reproduces the
official baseline, then autonomously iterates on features/hyperparameters to
beat it, on the KuaiRand-Pure recommendation benchmark (NDCG@10 / Recall@50,
click = positive).

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Get the data

Download and extract the real dataset:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzvf KuaiRand-Pure.tar.gz -C ./data/
```

This should leave you with `./data/KuaiRand-Pure/data/*.csv`.

**Don't have the real data yet / want to smoke-test the code first?**
Generate a small synthetic dataset with the *exact same file names and
columns* (meaningless labels, real schema):

```bash
python generate_sample_data.py --out_dir ./data/KuaiRand-Pure/data
```

## 3. Run

```bash
python main.py --data_dir ./data/KuaiRand-Pure/data --max_seconds 1800
```

This will:
1. Load the fixed train/val/test split (`data_loader.py`):
   - `log_standard_4_08_to_4_21_pure.csv` → train
   - `log_standard_4_22_to_5_08_pure.csv` first 50% (by time) → validation
   - `log_standard_4_22_to_5_08_pure.csv` last 50% (by time) → **held-out test — touched only once, at the very end**
2. Train the baseline-equivalent candidate.
3. Run the autonomous agent loop (`agent_loop.py`) over a pool of
   literature-informed feature/hyperparameter hypotheses, logging every
   iteration to `logs/run_log.jsonl` (hypothesis, code diff, metrics,
   error/recovery events — exactly what the challenge's Run-log
   requirements ask for).
4. Score the best candidate once on the test split and write
   `outputs/results_summary.json` with the delta vs. baseline.

## 4. Project layout

```
data_loader.py       # loads CSVs, builds the fixed train/val/test split
features.py          # feature-engineering pipeline (the "engineer features" stage)
model.py             # LightGBM training wrapper (the "train + tune" stage)
evaluate.py          # NDCG@10 / Recall@50 + delta-vs-baseline scoring
agent_loop.py         # the reflect+revise loop: propose -> train -> evaluate -> log
main.py              # orchestrates the full run end-to-end
generate_sample_data.py  # synthetic data generator for testing without the real download
requirements.txt
logs/run_log.jsonl        # per-iteration log (created on run)
outputs/results_summary.json  # final results table (created on run)
```

## 5. Where to extend

**Feature engineering** (`features.py`): add a new function with signature
`df -> df` and reference it in a `Candidate.feature_pipeline` in
`agent_loop.py`. Existing ones (engagement ratios, log-scaled activity
counters, time-of-day) are simple examples to build on — cross features,
target encoding, or embeddings all plug in the same way.

**Hypothesis generation** (`agent_loop.py: CANDIDATE_POOL`): this is
currently a fixed, hand-written pool standing in for what an LLM-driven
proposer would generate each round. To make hypothesis generation itself
autonomous/LLM-driven, replace the static list with a function that:
1. reads the previous iteration's metrics from `logs/run_log.jsonl`,
2. prompts an LLM for the next candidate (feature pipeline + hyperparams)
   and a stated hypothesis,
3. returns a new `Candidate`.
The rest of the loop (training, evaluation, robustness, logging,
convergence check) needs no changes.

**Model** (`model.py`): swap LightGBM for another algorithm by implementing
`train_model(train_df, val_df, feature_cols, ...) -> ModelWrapper` with the
same `.predict()` interface.

**Convergence rule**: `autonomous_agent_loop(..., epsilon=..., patience=...,
max_seconds=...)` — set these to match whatever ε / N / compute budget the
organizers publish in the official Starter Kit.

## 6. Known limitations (fill in after a real run)

- The `CANDIDATE_POOL` hypothesis set is hand-authored, not LLM-generated —
  true autonomy requires wiring in an LLM proposer as described above.
- No hyperparameter search beyond the 4 candidates in the pool; a real
  submission should widen this (e.g. Optuna) within the compute budget.
- Only `is_click` is modeled; KuaiRand's other 11 feedback signals
  (like, follow, long_view, etc.) are not used for multi-task learning,
  which the appendix flags as a legitimate way to fight label sparsity.
- Token/GPU-hour accounting (for the Feasibility & Practicality score) is
  not yet instrumented — add LLM call counting once an LLM proposer is wired
  in, and wrap `train_model` with GPU-time measurement if training moves to
  GPU-backed models.
