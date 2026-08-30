# FLIGHT — Autonomous ML Research Agent (TikTok TechJam Track 2)

An LLM-driven agent that autonomously runs the MLE loop — inspect data,
engineer features, train, evaluate, reflect, repeat — on the KuaiRand-Pure
recommendation benchmark, aiming to beat the organizers' official baseline
without human intervention.

## Task, as pinned by the organizers

| | |
|---|---|
| Task | **Within-user ranking** over logged impressions (not full-catalog retrieval) |
| Label | `long_view` (native 0/1 column) |
| Metrics | **GAUC** and **nDCG@5**; primary = mean of the two |
| Split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Convergence | ε = 0.002, N = 3; hard caps at 50 iterations / 6 h |

> **Note on the problem statement.** §2.3 of the TechJam doc says
> `NDCG@10 / Recall@50, click = positive`. That line is stale. §2.4, the
> judging criteria, and the shipped `official/evaluate.py` all specify
> GAUC / nDCG@5 on `long_view`, and the published baseline numbers are quoted
> in those metrics. **`official/evaluate.py` is the single source of truth.**

## Baselines to beat

Organizer-published, on the test split. These are *not* retrained by us —
improvement is always measured against these fixed numbers.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity floor) | 0.4996 | 0.4511 | 0.4753 |
| item popularity | 0.6308 | 0.5121 | 0.5715 |
| **FM (the one to beat)** | **0.6610** | **0.5282** | **0.5946** |
| oracle ceiling | 1.0000 | 0.7289 | 0.8645 |

**Read the ceiling, not 1.0.** 27.1% of test users have no positive label at
all (their nDCG is 0 for any model) and 9.2% are all-positive, so perfect
ranking tops out at primary 0.8645. The FM baseline already captures ~31% of
the attainable range; remaining headroom is ~0.27, not ~0.41.

## Setup

```bash
pip install -r requirements.txt

# ~47 MB
curl -L -o data/KuaiRand-Pure.tar.gz \
  https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf data/KuaiRand-Pure.tar.gz -C ./data/
```

## Reproduce our results

```bash
# 1. Sanity-check the harness. MUST print test primary ~0.4753 (+/- 0.001).
#    If it doesn't, the harness is broken -- fix that before anything else.
cd official && python baseline.py --model random --data_dir ../data/KuaiRand-Pure/data

# 2. Reproduce the official FM baseline (~1 min, CPU, no GPU).
cd official && python baseline.py --model fm --data_dir ../data/KuaiRand-Pure/data

# 3. Prove our pandas splits align row-for-row with the official loader.
#    This is what makes row_id in the submission valid.
python data_loader.py --data_dir ./data/KuaiRand-Pure/data --verify

# 4. Full agent run: iterate on train+valid, then score ONCE on test.
python main.py --data_dir ./data/KuaiRand-Pure/data

# 5. Validate the submission file.
cd official && python submit.py --check --split test ../submission.csv \
                 --data_dir ../data/KuaiRand-Pure/data
```

> **Windows note:** `submit.py` prints a `✓` on success, which crashes on a
> cp1252 console *after* all checks have passed. Prefix with
> `PYTHONIOENCODING=utf-8` to see the real output. A traceback on that print
> line means the file validated fine.

Verified on this machine: step 1 gives test primary **0.4757**; step 2 gives
valid **0.6015** / test **0.5953** against published 0.6016 / 0.5946
(σ = 0.0008 over 5 seeds); step 3 reports `sequence_matches: true` on all
three splits; step 5 reports `170,588 行` validated.

## Current result

| | GAUC | nDCG@5 | primary | vs baseline |
|---|---|---|---|---|
| Official FM baseline (test) | 0.6610 | 0.5282 | 0.5946 | — |
| **Agent, `switch_to_rank_xendcg` (test)** | **0.6650** | **0.5305** | **0.5978** | **+0.0032** ✅ |

Both metrics improve, not just one. +0.0032 is ~4σ of the baseline's 5-seed
σ=0.0008. Submission validated: `✓ 格式与对齐校验通过：170,588 行`.

**Resource usage** (feeds Feasibility & Practicality):

| | |
|---|---|
| Iterations | 5 of 50 (converged) |
| Agent wall-clock | 115 s |
| LLM tokens | 7,653 (6,943 in / 710 out) |
| GPU-hours | 0 — CPU only |
| **Manual interventions** | **0** |

### The run that got there

```
[0] starter (seed)                     0.5983  -0.0033
[1] lambdarank_objective_baseline      0.6043  +0.0027  BEATS   <- LLM
[2] lambdarank_tuned_depth_leaves      0.6041  +0.0025  BEATS   <- LLM
[3] lambda_ndcg_with_tuned_lambdarank  0.6040  +0.0024  BEATS   <- LLM
[4] switch_to_rank_xendcg              0.6061  +0.0045  BEATS   <- LLM
converged: no >0.002 gain in 3 consecutive iterations
```

The agent's own arc: it read the flat pointwise scores, hypothesised that a
*ranking* objective should beat pointwise logloss on ranking metrics, confirmed
it, tuned it to exhaustion, then generalised to the more nDCG-focused
`rank_xendcg` — the best result of the run. Nobody wrote those steps.

This independently confirms the organizers' #1 unexplored direction: the
bottleneck was the **objective**, not the feature set. Pointwise feature work
sat 0.004 *below* baseline; the objective switch alone cleared it.

## Project layout

```
official/              # Starter Kit, UNMODIFIED. Source of truth.
  evaluate.py          #   GAUC / nDCG@5. Do not edit.
  data.py              #   official splits, label, 5 encoded fields
  baseline.py          #   random / item-popularity / FM baselines
  baseline_scores.json #   published scores, seed variance, convergence rule
  submit.py            #   --make / --check / --score
  STARTER_KIT_README.md#   organizers' notes, incl. their own ablations (zh)

data_loader.py         # official splits + label, as pandas (for LightGBM path)
features.py            # feature engineering ("engineer features" stage)
model.py               # LightGBM wrapper ("train + tune" stage)
scoring.py             # official baseline refs, delta, convergence constants
agent_loop.py          # the reflect+revise loop -- THE PROJECT
main.py                # orchestrates a full run end-to-end

logs/run_log.jsonl         # per-iteration log (hypothesis/diff/metrics/errors)
outputs/results_summary.json
submission.csv
```

## What the organizers already tried (don't redo it)

From `official/STARTER_KIT_README.md`. These are measured, not guesses:

| Tried | Result |
|---|---|
| Adding CWM's full 13 feature fields | primary **0.5940** vs 0.5950 for the default 5. No gain |
| Embedding capacity k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887. Flat |

> The bottleneck is **not** features or capacity — `user_id × video_id`
> already absorbs most of the learnable signal.

**A trap worth internalizing:** ranking happens *within* a user, so anything
constant across that user's rows cannot reorder them. Pure user-side features
only pay off when crossed with item-side signal.

## Where the headroom is (organizer-ranked, untested by them)

1. **Change the loss.** Baseline is pointwise logloss, but GAUC/nDCG are
   ranking metrics. Pairwise (BPR) or listwise (softmax over the user's
   impressions) aligns objective with metric. Rated most likely to work.
2. **User behaviour sequences.** Entirely unused; DIN/SIM-style interest
   modelling is a blank field.
3. **Multi-task.** `is_click` / `is_like` / `is_follow` / `is_comment` /
   `is_forward` / `play_time_ms` as auxiliary heads.
4. **Watch-time modelling.** Censored regression on play time (CWM's idea) —
   watch time is truncated by video duration, so use a one-sided loss.
5. Swap the model (DeepFM / DCN / xDeepFM). Deprioritized: capacity isn't the bottleneck.
6. Time features and train/test drift (`hourmin`, `date`).
7. `log_random_4_22_to_5_08_pure.csv` — randomized-exposure log, usable as an
   unbiased validation set.

These are encoded in `agent_loop.py` as `KNOWN_DEAD_ENDS` and
`OPEN_DIRECTIONS`, ready to feed to the LLM proposer.

## Limitations / what we'd do next

- **Single seed.** The baseline's σ is 0.0008 over 5 seeds. Our +0.0032 is ~4σ
  so the sign is safe, but the writeup should quote a 3–5 seed mean.
- **The agent has not yet used its ability to write code.** `Candidate.code`
  supports agent-authored `fit`/`apply` features (unit-tested, leakage-safe),
  but every winning iteration so far changed only the objective. Directions 2–4
  (behaviour sequences, multi-task, watch-time) all need generated features —
  that is the biggest remaining upside.
- **Convergence fires early.** ε=0.002 with N=3 stops after ~5 iterations while
  50 are allowed. Since gains are now sub-ε but real, a wider exploration
  policy (or restarts from the best checkpoint) would use the budget properly.
- **The proposer cannot see the code it is editing.** It gets metrics, column
  names, and the direction list, but not `features.py` / `model.py` source.
  Feeding those in is the next step toward genuine code-level iteration.
- Model family is fixed to LightGBM; swapping in an FM/DeepFM that crosses
  `user_id × video_id` embeddings is untested.

## Team

| | |
|---|---|
| Pranav | Initial pipeline scaffold, agent loop structure, run-logging |
| Alfred | Starter-kit integration, official metric/split alignment, submission path |
| Arthur | — |
