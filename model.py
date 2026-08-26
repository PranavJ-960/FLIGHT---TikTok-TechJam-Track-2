"""
model.py
--------
Baseline ranking/CTR model for KuaiRand-Pure. Uses LightGBM binary
classification on `is_click`; predicted probability is used as the ranking
score for NDCG@10 / Recall@50.

This is deliberately swappable: `train_model` takes a `params` dict so an
agent iteration can change hyperparameters, and `ModelWrapper` is a thin
interface so the training stage in Figure 1 ("Train + tune") can be replaced
with a different algorithm without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

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


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "is_click",
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> ModelWrapper:
    """Train a LightGBM binary classifier with early stopping on val AUC."""
    params = {**DEFAULT_PARAMS, **(params or {})}

    train_set = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
    val_set = lgb.Dataset(val_df[feature_cols], label=val_df[label_col], reference=train_set)

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
