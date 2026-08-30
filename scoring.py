"""
scoring.py
----------
The single place that knows what "doing well" means, per the official
Starter Kit. Two jobs:

1. Hold the organizer-published baseline numbers (loaded from
   official/baseline_scores.json) so improvement is always measured against
   the *official* FM baseline -- never against a baseline we trained
   ourselves. The problem statement is explicit: "Beating this baseline is
   what counts -- not a baseline the team builds itself."

2. Hold the official convergence rule and compute budget.

The metric implementation itself lives in official/evaluate.py and must not
be modified -- import `evaluate` from there, never reimplement it.
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCORES_PATH = os.path.join(_HERE, "official", "baseline_scores.json")

with open(_SCORES_PATH, encoding="utf-8") as _fh:
    BASELINE_SCORES = json.load(_fh)

# --- Official reference points -------------------------------------------
# The row we must beat.
FM_BASELINE = BASELINE_SCORES["scores"]["fm_official"]
FM_BASELINE_VALID = FM_BASELINE["valid"]      # primary 0.6016
FM_BASELINE_TEST = FM_BASELINE["test"]        # primary 0.5946

# Sanity rungs. If our harness scores `random` far from this, the harness is
# broken and every other number is meaningless -- check this first.
RANDOM_BASELINE = BASELINE_SCORES["scores"]["random"]
POPULARITY_BASELINE = BASELINE_SCORES["scores"]["item_popularity"]

# Perfect ranking. nDCG@5 cannot reach 1.0 because 27.1% of test users have
# no positive label at all. Report progress as a fraction of THIS, not of 1.0.
ORACLE_CEILING = BASELINE_SCORES["scores"]["oracle_ceiling"]

# --- Official convergence rule and budget --------------------------------
EPSILON = BASELINE_SCORES["convergence_rule"]["epsilon"]   # 0.002 (~2.5 sigma)
PATIENCE_N = BASELINE_SCORES["convergence_rule"]["N"]      # 3
MAX_ITERATIONS = 50                                        # hard cap per run
MAX_SECONDS = 6 * 60 * 60                                  # 6h wall-clock backstop

METRIC_KEYS = ("GAUC", "nDCG@5")


def delta_vs_official(metrics: dict, split: str = "valid") -> dict:
    """
    delta(m) = score_agent(m) - score_baseline(m) for each scored metric,
    plus `score_dataset` = the equal-weighted mean of those deltas, exactly
    as defined in the Judging Criteria.

    `split` selects which published baseline row to compare against:
    'valid' while iterating, 'test' for the final one-shot report.
    """
    ref = FM_BASELINE_VALID if split == "valid" else FM_BASELINE_TEST
    deltas = {m: metrics[m] - ref[m] for m in METRIC_KEYS if m in metrics}
    deltas["score_dataset"] = sum(deltas.values()) / len(deltas) if deltas else 0.0
    deltas["primary_delta"] = metrics.get("primary", 0.0) - ref["primary"]
    return deltas


def progress_vs_oracle(metrics: dict, split: str = "valid") -> float:
    """
    Fraction of the *attainable* range captured, using the oracle ceiling as
    the denominator instead of 1.0. The FM baseline sits at ~0.307 of this.
    """
    rnd = RANDOM_BASELINE[split]["primary"]
    ceil_ = ORACLE_CEILING[split]["primary"]
    return (metrics["primary"] - rnd) / (ceil_ - rnd)


def summarize(metrics: dict, split: str = "valid") -> str:
    """One-line human-readable verdict for logs and stdout."""
    d = delta_vs_official(metrics, split)
    ref = FM_BASELINE_VALID if split == "valid" else FM_BASELINE_TEST
    verdict = "BEATS baseline" if d["primary_delta"] > 0 else "below baseline"
    return (
        f"{split}: GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} | "
        f"primary {metrics['primary']:.4f}  "
        f"(official FM {ref['primary']:.4f}, delta {d['primary_delta']:+.4f} -> {verdict}; "
        f"{progress_vs_oracle(metrics, split):.1%} of attainable range)"
    )
