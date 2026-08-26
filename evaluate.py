"""
evaluate.py
-----------
Ranking evaluation matching the challenge's metrics:
    NDCG@10 / Recall@50, click = positive.

Per user_id, videos are ranked by predicted score; relevance = is_click.
Metrics are computed per user (users with zero positive interactions in the
window are excluded, since recall/NDCG are undefined for them) and then
averaged across users -- this is the standard recsys evaluation protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def _ndcg_at_k(relevances_sorted_by_pred: np.ndarray, k: int) -> float:
    ideal = np.sort(relevances_sorted_by_pred)[::-1]
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return np.nan
    dcg = _dcg_at_k(relevances_sorted_by_pred, k)
    return dcg / idcg


def _recall_at_k(relevances_sorted_by_pred: np.ndarray, k: int) -> float:
    total_relevant = relevances_sorted_by_pred.sum()
    if total_relevant == 0:
        return np.nan
    return float(relevances_sorted_by_pred[:k].sum() / total_relevant)


def evaluate_ranking(
    df: pd.DataFrame,
    score_col: str = "pred",
    label_col: str = "is_click",
    group_col: str = "user_id",
    k_ndcg: int = 10,
    k_recall: int = 50,
) -> dict:
    """
    Returns:
        {
          "ndcg@10": float, "recall@50": float,
          "n_users_scored": int, "n_users_skipped_no_positive": int
        }
    """
    ndcgs, recalls = [], []
    skipped = 0

    for _, group in df.groupby(group_col, sort=False):
        ordered = group.sort_values(score_col, ascending=False)
        relevances = ordered[label_col].to_numpy(dtype=float)

        if relevances.sum() == 0:
            skipped += 1
            continue

        ndcgs.append(_ndcg_at_k(relevances, k_ndcg))
        recalls.append(_recall_at_k(relevances, k_recall))

    return {
        f"ndcg@{k_ndcg}": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
        f"recall@{k_recall}": float(np.nanmean(recalls)) if recalls else float("nan"),
        "n_users_scored": len(ndcgs),
        "n_users_skipped_no_positive": skipped,
    }


def compute_delta_vs_baseline(agent_metrics: dict, baseline_metrics: dict) -> dict:
    """
    delta(m) = score_agent(m) - score_baseline(m), for each metric m,
    plus the equal-weighted average delta ("score_dataset" in the spec).
    """
    metric_keys = [k for k in agent_metrics if k.startswith(("ndcg@", "recall@"))]
    deltas = {k: agent_metrics[k] - baseline_metrics[k] for k in metric_keys}
    deltas["score_dataset"] = float(np.mean(list(deltas.values())))
    return deltas
