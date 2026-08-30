"""
candidate.py
------------
A `Candidate` is one iteration's proposal. Crucially it can carry *code the
agent wrote*, not just a selection from a menu of pre-existing functions --
that is what makes this a research agent rather than a config sweeper.

Agent-authored code must define two functions:

    def fit(train_df):        # sees TRAIN ONLY -- prevents leakage
        return state          # any picklable object

    def apply(df, state):     # called on train, valid and test alike
        df["new_col"] = ...
        return df

New columns are detected by diffing the frame's columns before and after
`apply`, so the agent does not have to declare them. They are coerced to
numeric and NaN/inf-cleaned before training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from features import DEFAULT_FEATURE_PIPELINE, build_features


@dataclass
class Candidate:
    name: str
    hypothesis: str
    feature_pipeline: list[Callable] = field(
        default_factory=lambda: list(DEFAULT_FEATURE_PIPELINE)
    )
    param_overrides: dict = field(default_factory=dict)
    code: str | None = None          # agent-authored fit/apply source
    source: str = "static"           # "static" | "llm"

    def code_diff(self) -> dict:
        """What this iteration actually changed -- goes in the run log."""
        return {
            "feature_pipeline": [fn.__name__ for fn in self.feature_pipeline],
            "param_overrides": self.param_overrides,
            "code": self.code,
            "source": self.source,
        }


class CandidateCodeError(RuntimeError):
    """Agent-authored code failed to compile or ran incorrectly."""


def compile_candidate_code(code: str) -> dict:
    """Exec agent-authored source and return its namespace.

    The agent writes this code, so this is not a security boundary -- it is a
    correctness boundary. We check the contract is satisfied and let any
    exception propagate to the loop's try/except, which logs it as a failed
    iteration and moves on (the robustness requirement).
    """
    ns: dict = {"pd": pd, "np": np, "pandas": pd, "numpy": np}
    try:
        exec(compile(code, "<agent_candidate>", "exec"), ns)
    except Exception as exc:
        raise CandidateCodeError(f"candidate code failed to compile: {exc}") from exc

    for required in ("fit", "apply"):
        if required not in ns or not callable(ns[required]):
            raise CandidateCodeError(
                f"candidate code must define a callable `{required}(...)`; "
                f"got {sorted(k for k in ns if not k.startswith('__'))}"
            )
    return ns


def _sanitize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Make agent-authored columns safe for LightGBM: numeric, finite, filled."""
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        df[c] = s.fillna(0.0)
    return df


def build_candidate_frames(
    candidate: Candidate,
    train_raw: pd.DataFrame,
    *other_raws: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """
    Featurize train + any number of other splits identically.

    `fit` sees the TRAIN frame only; the resulting state is applied unchanged
    to every other split. That is what keeps target encoding and similar
    train-statistics tricks leakage-free.

    Returns (train_feat, [other_feats...], feature_cols).
    """
    train_feat, feature_cols = build_features(train_raw, candidate.feature_pipeline)
    others = [build_features(df, candidate.feature_pipeline)[0] for df in other_raws]

    if candidate.code:
        ns = compile_candidate_code(candidate.code)

        try:
            state = ns["fit"](train_feat)
        except Exception as exc:
            raise CandidateCodeError(f"fit() raised: {exc}") from exc

        before = set(train_feat.columns)
        try:
            train_feat = ns["apply"](train_feat, state)
            others = [ns["apply"](df, state) for df in others]
        except Exception as exc:
            raise CandidateCodeError(f"apply() raised: {exc}") from exc

        if not isinstance(train_feat, pd.DataFrame):
            raise CandidateCodeError("apply() must return a DataFrame")

        new_cols = [c for c in train_feat.columns if c not in before]
        if not new_cols:
            raise CandidateCodeError("apply() added no new columns -- nothing to test")

        missing = [c for df in others for c in new_cols if c not in df.columns]
        if missing:
            raise CandidateCodeError(
                f"apply() added {new_cols} to train but not to every split "
                f"(missing: {sorted(set(missing))})"
            )

        train_feat = _sanitize(train_feat, new_cols)
        others = [_sanitize(df, new_cols) for df in others]
        feature_cols = feature_cols + new_cols

    return train_feat, others, feature_cols
