"""Tune XGBoost (rank:ndcg) on the causal features — untuned so far (iteration M just
used reasonable defaults). Train/valid only.
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
from experiments.data_causal import encode_causal

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
XGB_DEFAULT = 0.6243


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def sample_params(rng):
    return {
        'eta': float(rng.choice([0.03, 0.05, 0.08, 0.1, 0.15, 0.2])),
        'max_depth': int(rng.choice([4, 5, 6, 8, 10])),
        'min_child_weight': int(rng.choice([5, 10, 20, 50, 100])),
        'subsample': float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        'colsample_bytree': float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        'lambda': float(rng.choice([0.0, 0.1, 1.0, 5.0, 10.0])),
        'alpha': float(rng.choice([0.0, 0.1, 1.0])),
        'gamma': float(rng.choice([0.0, 0.01, 0.1, 1.0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--n_trials', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    enc, feature_names = encode_causal(a.data_dir)
    Xtr, ytr, utr = build_matrix(enc, 'train')
    Xva, yva, uva = build_matrix(enc, 'valid')

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    utr_s = np.array([utr[i] for i in order_tr])
    uva_s = np.array([uva[i] for i in order_va])
    _, group_tr = np.unique(utr_s, return_counts=True)
    _, group_va = np.unique(uva_s, return_counts=True)

    # per-GROUP weight (1/group size) — XGBoost's ranking `weight` is one value per query
    # group, not per row (unlike LightGBM/CatBoost). Same intent as tune_lgb.py's per-row
    # weighting: GAUC/nDCG@5 average per USER, so equalize each user's total training
    # influence regardless of how many impressions they contributed.
    group_weight_tr = (1.0 / group_tr).astype(np.float32)

    dtrain = xgb.DMatrix(Xtr_s, label=ytr_s, feature_names=feature_names)
    dtrain.set_weight(group_weight_tr)
    dtrain.set_group(group_tr)
    dvalid = xgb.DMatrix(Xva_s, label=yva_s, feature_names=feature_names)
    dvalid.set_group(group_va)
    dva_full = xgb.DMatrix(Xva, feature_names=feature_names)

    rng = np.random.default_rng(a.seed)
    print(f"=== tuning XGBoost, {a.n_trials} trials ===")
    best, best_scores = None, None
    for t in range(a.n_trials):
        params = sample_params(rng)
        run_params = dict(params, objective='rank:ndcg', eval_metric='ndcg@5', seed=a.seed, verbosity=0)
        t0 = time.time()
        bst = xgb.train(run_params, dtrain, num_boost_round=1000, evals=[(dvalid, 'valid')],
                         early_stopping_rounds=30, verbose_eval=False)
        scores = bst.predict(dva_full, iteration_range=(0, bst.best_iteration + 1))
        m = evaluate(uva, yva, scores)
        dt = time.time() - t0
        flag = ""
        if best is None or m['primary'] > best['valid']['primary']:
            best = {'params': params, 'valid': m, 'best_iteration': bst.best_iteration}
            best_scores = scores
            flag = "  <- new best"
        print(f"  [{t+1:2d}/{a.n_trials}] primary={m['primary']:.4f} rounds={bst.best_iteration:3d} "
              f"({dt:.1f}s) {params}{flag}")

    print(f"\nbest: primary={best['valid']['primary']:.4f} "
          f"(delta vs baseline {best['valid']['primary']-BASELINE_VALID['primary']:+.4f}, "
          f"delta vs untuned {best['valid']['primary']-XGB_DEFAULT:+.4f})")
    print(f"best params: {best['params']}")

    np.save(os.path.join(a.out_dir, 'xgb_valid_scores.npy'), best_scores)
    np.save(os.path.join(a.out_dir, 'xgb_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'xgb_valid_labels.npy'), yva)
    with open('logs/iteration_P_xgb_tune.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'xgb_untuned': XGB_DEFAULT, 'best': best},
                   fh, indent=2, default=float)
    print("logged to logs/iteration_P_xgb_tune.json (and overwrote outputs/xgb_valid_scores.npy)")


if __name__ == '__main__':
    main()
