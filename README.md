# KuaiRand-Pure Autonomous ML Research Agent

TechJam Track 2: an autonomous agent that reproduces the official baseline, then
iterates — feature engineering, model architecture, training strategy, loss
function, and ensembling — to beat it on KuaiRand-Pure within-user ranking,
respecting the competition's convergence rule, compute caps, and hidden-test
isolation throughout.

**Final result (hidden test, one-time evaluation, explicitly requested): primary +0.0327
over baseline (0.6273 vs 0.5946), GAUC +0.0372, nDCG@5 +0.0281.** See [Results](#results)
below. This number came from `experiments/score_hidden_test.py` — a distinct, clearly-labeled
script that only ever runs on direct request (see [Hidden-test scoring](#hidden-test-scoring)).

## Task definition (fixed by the organizers — do not change)

| | |
|---|---|
| Task | Within-user ranking — rank each user's own impressions in the eval set; not full-catalog retrieval |
| Label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary = mean of both** |
| Split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Convergence rule (default) | ε = 0.002, N = 3 |
| Hard caps | 50 iterations / 6h wall-clock per benchmark |

See [`starter_kit/evaluate.py`](starter_kit/evaluate.py) — the scoring convention is fixed there and was never
modified (verified byte-identical against the organizer-provided kit).

## Setup

```bash
pip install numpy pandas lightgbm xgboost catboost torch
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzf KuaiRand-Pure.tar.gz -C ./data/
```

## Reproduce the result

The converged result comes from one bounded, autonomous, logged run of
[`experiments/orchestrator.py`](experiments/orchestrator.py), followed by a retrain of its
winning configuration to confirm the numbers — both train/valid only:

```bash
# 1. The autonomous run that found the winning config (already logged — see logs/orchestrator_run.json).
#    ε=0.002 default; N (patience) declared before this run as 10, per section 2.9.1's
#    team-declared-N allowance — the default N=3 converges prematurely (see Reflection below).
python3 experiments/orchestrator.py --wall_clock_budget_s 3600 --patience 10 --seed 1

# 2. Retrains the run's winning LightGBM+XGBoost+CatBoost blend and confirms the valid score.
#    Does not load, predict on, or evaluate the test split.
python3 experiments/final_submission.py
```

## Hidden-test scoring

`experiments/score_hidden_test.py` is the one place in this repo that loads the test split. It
is never called by `orchestrator.py`, `final_submission.py`, or any other script — it only runs
when a human explicitly invokes it by name:

```bash
python3 experiments/score_hidden_test.py
python3 starter_kit/submit.py --check --split test --data_dir ./data/KuaiRand-Pure/data submission_pure.csv
python3 starter_kit/submit.py --score --split test --data_dir ./data/KuaiRand-Pure/data submission_pure.csv
```

## Results

### KuaiRand-Pure (required benchmark)

| | GAUC | nDCG@5 | primary | Δ primary vs baseline |
|---|---|---|---|---|
| Official baseline (FM) — valid | 0.6674 | 0.5357 | 0.6016 | — |
| Official baseline (FM) — **test** | 0.6610 | 0.5282 | 0.5946 | — |
| Our model — valid | 0.7103 | 0.5596 | 0.6350 | +0.0334 |
| **Our model — test (scored once)** | **0.6982** | **0.5563** | **0.6273** | **+0.0327** |

Format and score independently re-verified against the official `starter_kit/submit.py --check`
and `--score` (not just our own `evaluate()` call) — both match exactly.

Model: LightGBM + XGBoost + CatBoost, all LambdaRank-family ranking losses (not pointwise
logloss), on 16 causally-engineered features (time-decayed Bayesian-smoothed interaction
rates, recency, session position, trending-rate deltas — see
[`experiments/data_causal.py`](experiments/data_causal.py)), each trained with per-user row
weighting (1/user-group-size — corrects for GAUC/nDCG@5 averaging per user while pointwise
training loss otherwise weights every row equally), blended lgb=0.2/xgb=0.5/catboost=0.3.
Exact config: [`logs/orchestrator_report.md`](logs/orchestrator_report.md) (the run that found
it) and [`logs/hidden_test_score.json`](logs/hidden_test_score.json) (the final retrain +
one-time test score).

### KuaiRand-1K (bonus benchmark)

Official baseline reproduced (valid primary 0.6451). XGBoost transfers cleanly from Pure's
tuned config (0.6757, +0.0306); LightGBM does not (0.6298, −0.0153 — Pure's deep-tree config
doesn't suit 1K's much larger per-user groups). CatBoost and KuaiRand-27K not attempted —
compute better spent solidifying Pure, per the rubric ("KuaiRand-Pure is required and
determines 100% of the primary score; 1K/27K are optional bonus only"). See
[`logs/iteration_U_kuairand1k.json`](logs/iteration_U_kuairand1k.json).

## Project layout

```
starter_kit/                  # official reference — untouched (verified byte-identical)
  data.py / evaluate.py / baseline.py / submit.py / ablation_features.py / baseline_scores.json

experiments/
  data_causal.py              # the winning feature set (causal, leakage-safe)
  data_seq.py / data_mt.py    # sequence-history and multi-task auxiliary-label loaders
  train_lgb.py / train_xgb.py / train_catboost.py / blend_multi.py   # the winning model line
  tune_lgb.py / tune_xgb.py / tune_catboost.py   # random-search tuning (sample_params reused
                                                    # by the orchestrator's own proposers)
  orchestrator.py             # the autonomous iterate→evaluate→decide→stop agent loop
  final_submission.py         # retrain the winning config, confirm valid score (train/valid
                               # only — does not touch the test split)
  score_hidden_test.py        # ONE-TIME, explicitly-requested only: scores test, writes
                               # submission_pure.csv. Never called by any other script.
  fm_torch.py / dcn_torch.py / deepfm_torch.py / bst_torch.py / mmoe_torch.py
                               # neural-net architectures explored (all underperformed the
                               # GBT blend — see Reflection)
  train_dcn_pairwise.py       # pairwise (BPR) ranking loss on the DCN backbone
  train_cwm.py                # from-scratch censored watch-time regression (not a clone of
                               # hyz20/CWM — different label formulation, no torch==1.6.0 dep)
  run_1k.py / run_1k_v2.py    # KuaiRand-1K bonus benchmark

logs/                          # every iteration's hypothesis/config/metrics — the run-log
                                # deliverable. orchestrator_run.json + orchestrator_report.md
                                # are the authoritative scored-run log.
data/KuaiRand-Pure/             # real dataset (gitignored, download per Setup above)
submission_pure.csv             # final hidden-test submission (row_id,user_id,video_id,score)
```

## Reflection: what worked, what didn't, what I'd do with more time

**What worked:** causal (leakage-safe, strictly-past-only) feature engineering — time-decayed
target-encoded rates, recency, trending-rate deltas — combined with per-user row weighting
(GAUC/nDCG@5 average per user; naive training loss doesn't, so a 200-impression user got 200×
the gradient influence of a 1-impression user despite counting equally in the score — fixing
this was the single biggest lever found all session) and a 3-way LightGBM/XGBoost/CatBoost
LambdaRank-family blend.

**What the organizers' own suggested directions turned up, tested honestly:**
1. *Pairwise/listwise loss* — already the default for the winning GBT line (LambdaRank/
   rank:ndcg/YetiRank). Separately retried as an explicit BPR loss on the DCN neural backbone
   (`train_dcn_pairwise.py`): 0.6144, actually *below* DCN's own pointwise-BCE version (0.6206)
   — noisier pair-sampling, not obviously better even though it's the closer theoretical match
   to the ranking metric.
2. *User history sequences* — tried two ways: DIN-style single-step attention (folded into the
   DCN/DeepFM backbones) and a real BST/SASRec-style self-attention Transformer over each
   user's actual chronological history (`train_bst.py`). The Transformer did *worse than
   baseline* (0.5803 vs 0.6015) — Pure's per-user histories are short and sparse, too little
   sequence for a full attention stack to learn from without overfitting.
3. *Multi-task auxiliary signals* — tried both a lightweight shared-embedding version and a
   proper gated MMoE (`train_mmoe.py`, `is_click`/`is_like`/`is_follow` as auxiliary tasks):
   0.6080, a marginal beat of baseline but well below the GBT blend — `is_like` (1.9% positive)
   and `is_follow` (0.1% positive) are too sparse to contribute much gradient signal.
4. *Censored watch-time regression* — not a clone of hyz20/CWM (different label formulation,
   would need `torch==1.6.0`); implemented the core idea from scratch instead
   (`train_cwm.py`): a Tobit-style custom LightGBM objective regressing
   `play_time_ms/duration_ms`, right-censored at 1.0 for looped plays, predicted ratio used as
   the ranking score. Result: 0.5486, *worse than baseline* — a continuous watch-ratio proxy
   doesn't track `long_view`-ranking as well as optimizing a ranking loss on the label directly.
5. *Other backbones (DeepFM/DCN/xDeepFM)* — DCN (0.6206) and DeepFM (0.6071) both built and
   tuned; both plateau below the GBT blend, consistent with the broader finding that GBTs
   outperform neural nets on tabular data at this row count. xDeepFM not attempted (diminishing
   expected return given the pattern above).

None of these closed the gap to the GBT blend — a genuinely negative, honestly-reported result
across nine independent neural-net/loss-function variants, not under-search.

**With more time:** (a) KuaiRand-27K — downloaded partway then deprioritized for time; the same
symlink trick that made 1K a zero-code-change bonus benchmark should work there too. (b) A
proper ESMM-style conditional combination (`P(click) × P(long_view|click)`) rather than MMoE's
shared-expert approach — abandoned mid-session because KuaiRand's tab/autoplay impressions mean
`is_click` doesn't cleanly gate `long_view` the way ESMM's e-commerce funnel assumes, but worth
verifying empirically rather than assuming. (c) A wider orchestrator proposer search (xDeepFM,
BST as first-class proposers, not just standalone scripts) so the autonomous run itself could
explore the full architecture space, not just GBT hyperparameters + blend weights.

## Autonomy / manual interventions

The **scored run** (`orchestrator.py`, logged in `logs/orchestrator_run.json`) executed fully
unattended: 14 iterations, 504.8s wall-clock, 0 manual interventions during execution, 0 errors.
The proposers' search *ranges* (LightGBM/XGBoost/CatBoost hyperparameter distributions) were
informed by earlier interactive research documented in `logs/iteration_{J,P,Q}_*_tune.json` —
disclosed here rather than presented as if the orchestrator found everything from a blank slate.

## Resource usage (Feasibility & Practicality)

| | Scored run (`orchestrator.py` + `score_hidden_test.py`) |
|---|---|
| Iterations | 14 / 50 cap |
| Wall-clock | 504.8s (orchestrator) + ~102s (final retrain + one-time test scoring) ≈ 607s, ≈0.17h — well under the 6h cap |
| GPU | 0 (LightGBM/XGBoost/CatBoost are CPU-only) |
| LLM calls inside the loop | 0 (pure programmatic weighted-sampling search, no LLM call in the proposer/decide loop itself) |
| Errors | 0 |

LLM token consumption is the cost of the *agent development session* that designed this
pipeline (feature engineering, architecture choices, the orchestrator itself) — not queryable
from inside the agent's own tool-use context; pull it from the Claude Code session's own usage/
cost display when compiling the Devpost submission.

## Known dead ends (from the organizers' own ablation, reproduced)

- Adding CWM's 13 static feature domains to the *baseline FM*: no gain (0.5940 vs 0.5950).
- Bigger baseline-FM embedding dims (k = 8/16/32): no gain (~0.589 flat).
- `user_id × video_id` interaction already captures most learnable signal; pure user-side
  features contribute nothing to within-user ranking (constant within a user's group).

## Team

Solo entry. All code, feature engineering, model/architecture decisions, and the autonomous
orchestrator were developed interactively with Claude Code (Anthropic) as the coding agent —
see [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md) for the full development-tools breakdown.

## History

An earlier iteration of this repo (commit `ceae8ad`) built a from-scratch pipeline against
NDCG@10/Recall@50 on `click`, which is **not** the actual scored metric — replaced 2026-08-28
after cross-checking the official starter kit.
