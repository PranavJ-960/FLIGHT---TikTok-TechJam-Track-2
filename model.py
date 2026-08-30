"""
model.py
--------
Training wrapper for KuaiRand-Pure. LightGBM on `long_view` (the official
relevance label); the predicted score is fed to official/evaluate.py
(GAUC / nDCG@5).

Two objectives are supported, and the choice is the agent's to make:

- `binary` (default)  -- pointwise logloss. Predicts P(long_view) per row.
- `lambdarank`        -- listwise. Optimizes ranking *within each user*, which
                         is what GAUC and nDCG@5 actually measure. The
                         organizers flag objective/metric mismatch as the most
                         promising unexplored direction.

For lambdarank LightGBM needs rows grouped contiguously by user and a `group`
array of per-user counts; that bookkeeping is handled here so a Candidate can
switch objectives with a single param override.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from official.data import LABEL as LABEL_COL

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}

RANKING_OBJECTIVES = {"lambdarank", "rank_xendcg"}


@dataclass
class ModelWrapper:
    booster: lgb.Booster | None = None
    feature_cols: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    best_iteration: int | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.booster.predict(
            df[self.feature_cols], num_iteration=self.best_iteration
        )


def _grouped(df: pd.DataFrame, feature_cols: list[str], label_col: str,
             group_col: str = "user_id"):
    """Sort rows so each user's impressions are contiguous, and return the
    per-user counts LightGBM's ranking objectives require."""
    order = np.argsort(df[group_col].to_numpy(), kind="stable")
    d = df.iloc[order]
    counts = d.groupby(group_col, sort=False).size().to_numpy()
    return d[feature_cols], d[label_col], counts


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = LABEL_COL,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> ModelWrapper:
    """Train LightGBM with early stopping on the validation split."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    objective = params.get("objective", "binary")

    if objective in RANKING_OBJECTIVES:
        # Ranking objectives need contiguous groups + counts, and their own metric.
        params.setdefault("eval_at", [5])
        if params.get("metric") in (None, "auc"):
            params["metric"] = "ndcg"

        Xtr, ytr, gtr = _grouped(train_df, feature_cols, label_col)
        Xva, yva, gva = _grouped(val_df, feature_cols, label_col)
        train_set = lgb.Dataset(Xtr, label=ytr, group=gtr)
        val_set = lgb.Dataset(Xva, label=yva, group=gva, reference=train_set)
    else:
        train_set = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
        val_set = lgb.Dataset(val_df[feature_cols], label=val_df[label_col],
                              reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return ModelWrapper(
        booster=booster,
        feature_cols=feature_cols,
        params=params,
        best_iteration=booster.best_iteration,
    )
