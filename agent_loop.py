"""
agent_loop.py
-------------
Implements the MLE iteration loop (read problem -> inspect data -> engineer
features -> train+tune -> evaluate -> reflect+revise) as an autonomous,
logged control loop, per the challenge's Task Requirements and Run-log
requirements.

Design:
- Each iteration is a `Candidate`: a named feature pipeline + a hyperparameter
  dict. This is the unit the agent proposes, tests, and reflects on.
- `CANDIDATE_POOL` encodes a small set of literature-informed hypotheses
  (multi-task-style engagement features, wider trees, feature-fraction
  regularization, etc.) that stand in for what an LLM would propose. Replace
  `propose_next_candidate` with an actual LLM call to make hypothesis
  generation itself model-driven -- the rest of the loop (robustness,
  logging, convergence check) does not need to change.
- Robustness: every iteration is wrapped in try/except. A failed iteration is
  logged with its error and the loop continues with the next candidate
  instead of crashing (Task Requirement 3).
- Convergence: stops when validation score_dataset has not improved by more
  than `epsilon` over the last `patience` iterations, or when
  `max_iterations` / `max_seconds` is hit -- whichever comes first.
- Every iteration is appended as one JSON line to `logs/run_log.jsonl`,
  containing: hypothesis, code diff (the candidate's config), metrics, and
  any error/recovery event, as required by section "Run-log requirements".
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Callable

import pandas as pd

from evaluate import evaluate_ranking, compute_delta_vs_baseline
from features import (
    DEFAULT_FEATURE_PIPELINE,
    add_engagement_ratio_features,
    add_user_activity_features,
    add_time_features,
    build_features,
)
from model import train_model, DEFAULT_PARAMS


@dataclass
class Candidate:
    name: str
    hypothesis: str
    feature_pipeline: list[Callable] = field(default_factory=lambda: list(DEFAULT_FEATURE_PIPELINE))
    param_overrides: dict = field(default_factory=dict)


# A small, literature-informed pool of hypotheses standing in for what an
# LLM-driven proposer would generate each round (see module docstring).
CANDIDATE_POOL: list[Candidate] = [
    Candidate(
        name="baseline",
        hypothesis="Establish the official-baseline-equivalent pipeline: raw "
                   "numeric + categorical features, default LightGBM params.",
        feature_pipeline=[add_time_features],
        param_overrides={},
    ),
    Candidate(
        name="engagement_ratios",
        hypothesis="CTR correlates with prior engagement rates (play_rate, "
                   "completion_rate) more than raw counts, per standard CTR "
                   "feature-engineering practice (crossing engagement signals). "
                   "Add ratio features on top of the baseline.",
        feature_pipeline=[add_time_features, add_engagement_ratio_features],
        param_overrides={},
    ),
    Candidate(
        name="engagement_plus_user_activity",
        hypothesis="Log-scaling heavy-tailed user activity counters "
                   "(follow/fans/friend counts, register_days) should help "
                   "the tree model split more evenly across active/inactive "
                   "users, on top of engagement ratios.",
        feature_pipeline=[add_time_features, add_engagement_ratio_features, add_user_activity_features],
        param_overrides={},
    ),
    Candidate(
        name="deeper_trees_more_regularization",
        hypothesis="With the fuller feature set, allow more model capacity "
                   "(num_leaves) but add feature_fraction regularization to "
                   "avoid overfitting to any single engineered feature.",
        feature_pipeline=[add_time_features, add_engagement_ratio_features, add_user_activity_features],
        param_overrides={"num_leaves": 127, "feature_fraction": 0.7, "learning_rate": 0.03},
    ),
]


def run_iteration(
    candidate: Candidate,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    label_col: str = "is_click",
) -> dict:
    """Run one full iteration: build features, train, evaluate. Returns a
    result dict; raises on unrecoverable errors (caller handles logging)."""
    train_feat, feature_cols = build_features(train_raw, candidate.feature_pipeline)
    val_feat, _ = build_features(val_raw, candidate.feature_pipeline)

    model = train_model(
        train_feat, val_feat, feature_cols,
        label_col=label_col,
        params=candidate.param_overrides,
    )

    val_feat = val_feat.copy()
    val_feat["pred"] = model.predict(val_feat)
    metrics = evaluate_ranking(val_feat, score_col="pred", label_col=label_col)

    return {
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model": model,
    }


class RunLogger:
    """Writes one JSON line per iteration to logs/run_log.jsonl, matching the
    challenge's required fields: hypothesis, code diff, metrics, error/recovery."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._n_manual_interventions = 0  # this script runs fully autonomously -> 0

    def log(self, record: dict):
        record["timestamp"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def summary(self):
        return {"manual_interventions": self._n_manual_interventions}


def autonomous_agent_loop(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    baseline_metrics: dict,
    log_path: str = "logs/run_log.jsonl",
    candidates: list[Candidate] | None = None,
    epsilon: float = 0.001,
    patience: int = 2,
    max_seconds: float = 1800.0,
) -> dict:
    """
    Runs the reflect+revise loop over `candidates` (default: CANDIDATE_POOL),
    logging every iteration, and stops at convergence (score_dataset hasn't
    improved by > epsilon over the last `patience` iterations) or when
    max_seconds is exceeded.

    Returns the best candidate's result plus the full history.
    """
    candidates = candidates if candidates is not None else CANDIDATE_POOL
    logger = RunLogger(log_path)

    history = []
    best_result = None
    best_score = float("-inf")
    no_improve_streak = 0
    start = time.time()

    for i, candidate in enumerate(candidates):
        if time.time() - start > max_seconds:
            logger.log({"event": "stop", "reason": "max_seconds_budget_reached", "iteration": i})
            break

        record = {
            "iteration": i,
            "candidate_name": candidate.name,
            "hypothesis": candidate.hypothesis,
            "code_diff": {
                "feature_pipeline": [fn.__name__ for fn in candidate.feature_pipeline],
                "param_overrides": candidate.param_overrides,
            },
        }

        try:
            result = run_iteration(candidate, train_raw, val_raw)
            deltas = compute_delta_vs_baseline(result["metrics"], baseline_metrics)
            score = deltas["score_dataset"]

            record["metrics"] = result["metrics"]
            record["delta_vs_baseline"] = deltas
            record["status"] = "success"
            history.append(record)
            logger.log(record)

            if score > best_score + epsilon:
                best_score = score
                best_result = {**result, "candidate": candidate, "delta_vs_baseline": deltas}
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            if no_improve_streak >= patience:
                logger.log({
                    "event": "stop", "reason": "converged",
                    "iteration": i, "no_improve_streak": no_improve_streak,
                })
                break

        except Exception as exc:  # noqa: BLE001 - deliberately broad: robustness requirement
            record["status"] = "error"
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
            record["recovery_action"] = "skip_candidate_continue_loop"
            history.append(record)
            logger.log(record)
            continue

    return {
        "best_result": best_result,
        "history": history,
        "run_summary": logger.summary(),
    }
