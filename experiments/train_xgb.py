"""Iteration M: XGBoost (rank:ndcg) on the same causal features LightGBM uses — a
genuinely different tree-boosting implementation (different regularization, different
split-finding), tried because the FM+LightGBM prediction blend failed (FM's errors
weren't complementary): a different GBT library has a better shot at decorrelated
errors than a weaker model of the same general kind. Train/valid only.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LGB_BEST = 0.6272   # 5-seed bagged, tuned (iteration J/K)


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


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
    Xtr, ytr, utr = build_matrix(enc, 'train')
    Xva, yva, uva = build_matrix(enc, 'valid')

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    utr_s = np.array([utr[i] for i in order_tr])
    uva_s = np.array([uva[i] for i in order_va])
    group_tr = np.unique(utr_s, return_counts=True)[1]
    group_va = np.unique(uva_s, return_counts=True)[1]

    group_weight_tr = (1.0 / group_tr).astype(np.float32)

    dtrain = xgb.DMatrix(Xtr_s, label=ytr_s, feature_names=feature_names)
    dtrain.set_weight(group_weight_tr)
    dtrain.set_group(group_tr)
    dvalid = xgb.DMatrix(Xva_s, label=yva_s, feature_names=feature_names)
    dvalid.set_group(group_va)

    params = {
        'objective': 'rank:ndcg', 'eval_metric': 'ndcg@5',
        'eta': 0.2, 'max_depth': 4, 'min_child_weight': 100,
        'subsample': 0.7, 'colsample_bytree': 0.8, 'lambda': 5.0,
        'alpha': 1.0, 'gamma': 1.0,
        'seed': a.seed, 'verbosity': 0,
    }
    print("\n=== iteration M: XGBoost rank:ndcg ===")
    t0 = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, 'valid')],
                     early_stopping_rounds=30, verbose_eval=False)
    print(f"  trained {bst.best_iteration} rounds in {time.time()-t0:.1f}s")

    dva_full = xgb.DMatrix(Xva, feature_names=feature_names)
    scores_va = bst.predict(dva_full, iteration_range=(0, bst.best_iteration + 1))
    metrics = evaluate(uva, yva, scores_va)
    print(f"  -> valid GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
          f"primary {metrics['primary']:.4f} | delta vs baseline "
          f"{metrics['primary']-BASELINE_VALID['primary']:+.4f} | delta vs LightGBM-best "
          f"{metrics['primary']-LGB_BEST:+.4f}")
    importance = bst.get_score(importance_type='gain')
    print("  feature importances (gain):", dict(sorted(importance.items(), key=lambda kv: -kv[1])[:10]))

    np.save(os.path.join(a.out_dir, 'xgb_valid_scores.npy'), scores_va)
    np.save(os.path.join(a.out_dir, 'xgb_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'xgb_valid_labels.npy'), yva)
    with open('logs/iteration_M_xgb.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'lgb_best': LGB_BEST, 'params': params,
                    'best_iteration': bst.best_iteration, 'valid': metrics,
                    'feature_importance_gain': importance}, fh, indent=2, default=float)
    print("logged to logs/iteration_M_xgb.json")


if __name__ == '__main__':
    main()
