# Project Description — KuaiRand-Pure Autonomous ML Research Agent

*(Written for the Devpost submission — Track 2: Autonomous ML Research Agent for
Recommender Systems)*

## How the solution addresses the problem statement

The challenge asks for an autonomous agent that runs the full MLE iteration loop
(read the problem → inspect data → engineer features → train & tune → evaluate → reflect
& revise) on KuaiRand-Pure's within-user ranking task (`long_view`, scored by
GAUC/nDCG@5), respecting the hidden-test boundary and the 50-iteration/6h compute cap.

This solution has two layers:

1. **`experiments/orchestrator.py`** — the actual autonomous agent. A pure-Python
   iterate → evaluate → decide → stop loop: each iteration is selected by adaptive,
   cost- and success-weighted sampling over five proposer types (LightGBM trial, XGBoost
   trial, CatBoost trial, feature-variant trial, blend trial), evaluated on validation only,
   logged incrementally (crash-safe), and the loop stops itself via the organizers'
   convergence rule (or a team-declared N, fixed before the run and disclosed) or the hard
   caps — whichever comes first. This is the process that is actually scored: one
   unattended, 14-iteration, 504.8-second run that reached a validation primary of 0.6350
   (+0.0334 over baseline), with zero manual interventions during execution and zero errors.

2. **`experiments/final_submission.py`** retrains that run's winning configuration and
   confirms the validation score. It does not touch the hidden test split — generating the
   actual scored submission CSV is a separate, explicitly-requested one-time step, not run
   as a default part of "produce the deliverables" (this repo treats hidden-test contact as
   something that must be asked for directly, every time, not inferred).

The proposers' search spaces were shaped by an earlier, broader research phase (feature
engineering, loss-function comparisons, five neural-architecture variants) that touched
every stage of the pipeline — feature engineering, training strategy, model architecture,
and the evaluation loop itself — not just model swaps, addressing the Innovation &
Problem Insight criterion directly. That phase is disclosed as prior research informing the
orchestrator's design, not folded into the scored run's own resource/iteration count.

## Development tools

- **Claude Code** (Anthropic) — the coding agent used to develop, test, and iterate this
  entire pipeline interactively (VS Code integration).
- Standard shell/Python tooling — no notebook environment; all experiments run as
  standalone `python3` scripts for reproducibility and crash-isolation.

## APIs used

None — the pipeline is self-contained. No external LLM API calls are made during model
training or evaluation; all "agent" behavior (iterate/evaluate/decide/stop) is
programmatic, not LLM-driven, keeping the scored run's own LLM-token cost at zero.

## Libraries and frameworks used

- **NumPy / pandas** — feature engineering, causal (leakage-safe) rate/recency computation.
- **LightGBM, XGBoost, CatBoost** — the three gradient-boosted ranking models
  (LambdaRank / `rank:ndcg` / YetiRank objectives) that make up the winning blend.
- **PyTorch** (with Apple Silicon MPS acceleration where available) — the neural-net
  architectures explored: FM, DCN, DeepFM, a BST/SASRec-style Transformer, and an MMoE
  multi-task model. All five were tuned and honestly evaluated; none beat the GBT blend
  (see the Reflection section of `README.md`).

## Datasets and assets used

- **KuaiRand-Pure** (required benchmark) — 1.14M train / 124.9K valid / 170.6K test
  interactions, from https://kuairand.com (Zenodo record 10439422). Used strictly per the
  organizers' fixed date-based split; the randomized-exposure log
  (`log_random_4_22_to_5_08_pure.csv`) was never used for training, matching the rules'
  data-clarification note.
- **KuaiRand-1K** (bonus benchmark) — attempted as its own independent benchmark (never
  mixed into Pure's training, per the rules) via a symlink trick that lets the unmodified
  official starter kit and this repo's own scripts run against it with zero code changes.
- No external training data and no pretrained weights were used anywhere in the pipeline.
