"""
agent_loop.py
-------------
The MLE iteration loop (read problem -> inspect data -> engineer features ->
train+tune -> evaluate -> reflect+revise) as an autonomous, logged control
loop, per the challenge's Task Requirements and Run-log requirements.

Scoring is delegated entirely to official/evaluate.py (GAUC / nDCG@5 on
`long_view`) and improvement is measured against the ORGANIZER-PUBLISHED FM
baseline via scoring.py -- never against a baseline we trained ourselves.

Design:
- Each iteration is a `Candidate` (candidate.py): a stated hypothesis plus the
  code and/or LightGBM parameters that test it. Candidates can carry
  agent-authored `fit`/`apply` source, so an iteration is a real code change.
- `proposer.py` generates candidates -- LLM-driven when a key is present,
  a fixed pool otherwise.
- Robustness: every iteration is wrapped in try/except. A failed iteration is
  logged with its error and recovery action; the loop continues.
- Convergence: official rule -- stop when validation primary has not improved
  by more than EPSILON over the last PATIENCE_N iterations, or at the
  50-iteration cap, or at the 6h ceiling, whichever comes first.
"""

from __future__ import annotations

import json
import time
import traceback

import pandas as pd

from official.data import LABEL as LABEL_COL
from official.evaluate import evaluate
from features import (
    add_engagement_ratio_features,
    add_user_activity_features,
    add_time_features,
)
from candidate import Candidate, build_candidate_frames
from model import train_model
import scoring


# --- What the organizers already tested, so the agent does not redo it ----
# Straight from official/STARTER_KIT_README.md. Fed to the LLM proposer.
KNOWN_DEAD_ENDS = [
    "Adding more static feature fields: the organizers wired in CWM's full 13 "
    "fields and got primary 0.5940 vs 0.5950 for the default 5 fields. No gain.",
    "More model capacity: embedding k=8/16/32 gave 0.5895/0.5902/0.5887. Flat. "
    "The bottleneck is NOT features or capacity -- user_id x video_id already "
    "absorbs most of the learnable signal.",
    "Pure user-side features cannot help on their own: ranking is WITHIN a "
    "user, so any term constant across that user's rows cannot reorder them. "
    "User features only pay off when crossed with item-side signal.",
]

# Organizer-ranked unexplored directions, most promising first.
OPEN_DIRECTIONS = [
    "Change the loss. Baseline is pointwise logloss but GAUC/nDCG are ranking "
    "metrics. Pairwise (BPR) or listwise (softmax over the user's impressions) "
    "aligns the objective with the metric. Organizers rate this most likely. "
    "In this harness: param_overrides {'objective': 'lambdarank'}.",
    "User behaviour sequences. Entirely unused today; DIN/SIM-style interest "
    "modelling over each user's history is a blank field.",
    "Multi-task. is_click / is_like / is_follow / is_comment / is_forward / "
    "play_time_ms as auxiliary heads on the long_view main task.",
    "Watch-time modelling. Censored regression on play_time (CWM's approach) -- "
    "watch time is truncated by video duration, so use a one-sided loss.",
    "Swap the model (DeepFM / DCN / xDeepFM). Deprioritised: capacity is not "
    "the bottleneck.",
    "Time features and train/test distribution drift (hourmin, date).",
    "Unbiased validation: log_random_4_22_to_5_08_pure.csv is a randomised-"
    "exposure log usable as a bias-free validation set.",
]


# Seed candidates. With an LLM proposer only the first is used, as a scored
# reference point; without one, the loop walks the whole list.
CANDIDATE_POOL: list[Candidate] = [
    Candidate(
        name="starter",
        hypothesis="Stand up a working end-to-end pipeline on the official "
                   "split and label (long_view), scored by official/evaluate.py.",
        feature_pipeline=[add_time_features],
        param_overrides={},
    ),
    Candidate(
        name="engagement_ratios",
        hypothesis="Item-side engagement priors (play_rate, completion_rate) "
                   "vary within a user's impression list, so unlike user-side "
                   "counters they CAN reorder it.",
        feature_pipeline=[add_time_features, add_engagement_ratio_features],
        param_overrides={},
    ),
    Candidate(
        name="lambdarank",
        hypothesis="GAUC and nDCG are ranking metrics but the model trains "
                   "pointwise logloss. Switching to a listwise objective "
                   "grouped per user aligns training with evaluation.",
        feature_pipeline=[add_time_features, add_engagement_ratio_features],
        param_overrides={"objective": "lambdarank"},
    ),
    Candidate(
        name="deeper_trees_more_regularization",
        hypothesis="More capacity with feature_fraction regularization, to "
                   "check whether the organizers' flat-capacity finding for FM "
                   "also holds for GBDTs.",
        feature_pipeline=[add_time_features, add_engagement_ratio_features,
                          add_user_activity_features],
        param_overrides={"num_leaves": 127, "feature_fraction": 0.7,
                         "learning_rate": 0.03},
    ),
]


def score_frame(df: pd.DataFrame, preds) -> dict:
    """Score a frame with the OFFICIAL metric. Never reimplement this."""
    return evaluate(
        df["user_id"].tolist(),
        df[LABEL_COL].astype(int).tolist(),
        list(preds),
    )


def run_iteration(candidate: Candidate, train_raw, val_raw,
                  label_col: str = LABEL_COL) -> dict:
    """One full iteration: build features (incl. agent code), train, evaluate."""
    train_feat, (val_feat,), feature_cols = build_candidate_frames(
        candidate, train_raw, val_raw
    )
    if not feature_cols:
        raise ValueError("candidate produced no feature columns")

    model = train_model(train_feat, val_feat, feature_cols,
                        label_col=label_col, params=candidate.param_overrides)

    preds = model.predict(val_feat)
    metrics = score_frame(val_feat, preds)
    return {"feature_cols": feature_cols, "metrics": metrics, "model": model}


class RunLogger:
    """One JSON line per iteration: hypothesis, code diff, metrics,
    error/recovery -- the challenge's Run-log requirements."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.manual_interventions = 0   # fully autonomous run -> 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def log(self, record: dict):
        record["timestamp"] = time.time()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def summary(self):
        return {
            "manual_interventions": self.manual_interventions,
            "llm_input_tokens": self.usage["input_tokens"],
            "llm_output_tokens": self.usage["output_tokens"],
            "llm_total_tokens": self.usage["input_tokens"] + self.usage["output_tokens"],
        }


def autonomous_agent_loop(
    train_raw,
    val_raw,
    log_path: str = "logs/run_log.jsonl",
    proposer=None,
    epsilon: float = scoring.EPSILON,                # 0.002
    patience: int = scoring.PATIENCE_N,              # 3
    max_iterations: int = scoring.MAX_ITERATIONS,    # 50
    max_seconds: float = scoring.MAX_SECONDS,        # 6h
) -> dict:
    """
    Reflect+revise loop. Selection and convergence both track validation
    PRIMARY (mean of GAUC and nDCG@5) -- the official ranking quantity.
    Returns the validation-best result plus full history.
    """
    if proposer is None:
        from proposer import make_proposer
        proposer = make_proposer(seed_pool=CANDIDATE_POOL)

    logger = RunLogger(log_path)
    columns = [c for c in train_raw.columns]

    history, best_result = [], None
    best_primary = float("-inf")
    no_improve_streak = 0
    start = time.time()

    for i in range(max_iterations):
        if time.time() - start > max_seconds:
            logger.log({"event": "stop", "reason": "max_seconds_budget_reached",
                        "iteration": i})
            break

        # --- propose -------------------------------------------------------
        try:
            candidate = proposer.propose(history, columns, i)
        except Exception as exc:
            logger.log({"iteration": i, "status": "error", "stage": "propose",
                        "error": str(exc), "traceback": traceback.format_exc(),
                        "recovery_action": "abort_loop_keep_best"})
            print(f"  [{i}] proposer failed ({exc}) -- stopping with best so far")
            break

        if candidate is None:
            logger.log({"event": "stop", "reason": "proposer_exhausted",
                        "iteration": i})
            break

        record = {
            "iteration": i,
            "candidate_name": candidate.name,
            "hypothesis": candidate.hypothesis,
            "code_diff": candidate.code_diff(),
        }

        # --- test ----------------------------------------------------------
        try:
            result = run_iteration(candidate, train_raw, val_raw)
            metrics = result["metrics"]
            deltas = scoring.delta_vs_official(metrics, split="valid")

            record["metrics"] = metrics
            record["delta_vs_official_baseline"] = deltas
            record["status"] = "success"
            history.append(record)
            logger.log(record)
            print(f"  [{i}] {candidate.name}: {scoring.summarize(metrics, 'valid')}")

            # Two SEPARATE questions, and conflating them loses good models:
            #   selection  -- is this the validation-best so far? (strict >)
            #                 the spec submits "the validation-best checkpoint"
            #   convergence-- did it improve by more than epsilon? (the official
            #                 ε=0.002 / N=3 plateau rule)
            # A +0.001 gain is a better checkpoint AND a converging run.
            improved_materially = metrics["primary"] > best_primary + epsilon

            if metrics["primary"] > best_primary:
                best_primary = metrics["primary"]
                best_result = {**result, "candidate": candidate,
                               "delta_vs_official_baseline": deltas}

            no_improve_streak = 0 if improved_materially else no_improve_streak + 1

            if no_improve_streak >= patience:
                logger.log({"event": "stop", "reason": "converged", "iteration": i,
                            "epsilon": epsilon, "N": patience,
                            "no_improve_streak": no_improve_streak})
                print(f"  converged: no >{epsilon} gain in {patience} "
                      f"consecutive iterations")
                break

        except Exception as exc:  # noqa: BLE001 -- broad on purpose: robustness requirement
            record["status"] = "error"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            record["recovery_action"] = "log_error_and_propose_next_candidate"
            history.append(record)
            logger.log(record)
            print(f"  [{i}] {candidate.name}: FAILED ({str(exc)[:120]}) -- recovering")
            continue
        finally:
            logger.usage = getattr(proposer, "usage", logger.usage)

    return {
        "best_result": best_result,
        "history": history,
        "run_summary": {**logger.summary(), "proposer": getattr(proposer, "source", "?")},
        "elapsed_seconds": time.time() - start,
    }
