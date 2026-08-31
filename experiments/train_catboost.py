"""Iteration N: CatBoost (YetiRank) on the same causal features — a third, distinctly
different GBT implementation (ordered boosting, its own categorical handling) to test
for complementary errors alongside LightGBM and XGBoost. Train/valid only.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS, NUMERIC_FIELDS

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LGB_BEST = 0.6272


def build_frame(enc, split):
    """CatBoost needs actual int/str dtype columns for categoricals — a homogeneous
    float ndarray with cat_features indices errors out."""
    d = enc[split]
    df = pd.DataFrame(d['num'], columns=NUMERIC_FIELDS)
    df['dow'] = d['dow'].astype(str)
    for i, c in enumerate(CAT_FIELDS):
        df[c] = d['cat'][:, i].astype(str)
    return df, d['y'], d['users']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("=== building causal features ===")
    enc, feature_names = encode_causal(a.data_dir)
    Xtr, ytr, utr = build_frame(enc, 'train')
    Xva, yva, uva = build_frame(enc, 'valid')
    cat_cols = ['dow'] + CAT_FIELDS

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr.iloc[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva.iloc[order_va], yva[order_va]
    group_tr = np.array([utr[i] for i in order_tr])
    group_va = np.array([uva[i] for i in order_va])

    group_map = dict(zip(*np.unique(group_tr, return_counts=True)))
    gw_tr = np.array([1.0 / group_map[u] for u in group_tr], dtype=np.float32)

    train_pool = Pool(Xtr_s, label=ytr_s, group_id=group_tr, group_weight=gw_tr, cat_features=cat_cols)
    valid_pool = Pool(Xva_s, label=yva_s, group_id=group_va, cat_features=cat_cols)

    params = {
        'loss_function': 'YetiRank', 'eval_metric': 'NDCG:top=5',
        'learning_rate': 0.08, 'depth': 5, 'l2_leaf_reg': 1.0,
        'subsample': 0.6, 'rsm': 0.8, 'random_strength': 5.0,
        'iterations': 1000, 'random_seed': a.seed, 'verbose': False,
        'early_stopping_rounds': 30, 'thread_count': -1,
    }
    print("\n=== iteration N: CatBoost YetiRank ===")
    t0 = time.time()
    model = CatBoost(params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    print(f"  trained {model.get_best_iteration()} rounds in {time.time()-t0:.1f}s")

    valid_pool_full = Pool(Xva, cat_features=cat_cols)
    scores_va = model.predict(valid_pool_full)
    metrics = evaluate(uva, yva, scores_va)
    print(f"  -> valid GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
          f"primary {metrics['primary']:.4f} | delta vs baseline "
          f"{metrics['primary']-BASELINE_VALID['primary']:+.4f} | delta vs LightGBM-best "
          f"{metrics['primary']-LGB_BEST:+.4f}")
    importance = dict(zip(Xtr.columns, model.get_feature_importance(train_pool).tolist()))
    print("  feature importances:", dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10]))

    np.save(os.path.join(a.out_dir, 'catboost_valid_scores.npy'), scores_va)
    np.save(os.path.join(a.out_dir, 'catboost_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'catboost_valid_labels.npy'), yva)
    with open('logs/iteration_N_catboost.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'lgb_best': LGB_BEST, 'params': params,
                    'best_iteration': model.get_best_iteration(), 'valid': metrics,
                    'feature_importance': importance}, fh, indent=2, default=float)
    print("logged to logs/iteration_N_catboost.json")


if __name__ == '__main__':
    main()
