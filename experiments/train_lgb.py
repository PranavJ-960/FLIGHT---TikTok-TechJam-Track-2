"""Iteration H: LightGBM LambdaRank on causal features (temporal + time-decayed
target-encoded rates). Standalone process — kept separate from torch/MPS code
because running lightgbm and torch in the same process deadlocks on this machine
(dual-OpenMP/threading conflict between lightgbm's native threads and torch's MPS
backend, reproduced twice). Train/valid only — test is never unpacked/evaluated.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
FM_ENSEMBLE_VALID_PRIMARY = 0.6021


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("=== building causal features ===")
    t0 = time.time()
    enc, feature_names = encode_causal(a.data_dir)
    print(f"  built in {time.time()-t0:.1f}s, features: {feature_names}")

    Xtr, ytr, utr = build_matrix(enc, 'train')
    Xva, yva, uva = build_matrix(enc, 'valid')
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    utr_s = [utr[i] for i in order_tr]
    uva_s = [uva[i] for i in order_va]
    group_tr = np.unique(utr_s, return_counts=True)[1]
    group_va = np.unique(uva_s, return_counts=True)[1]

    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=group_tr, feature_name=feature_names,
                          categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)

    params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [5],
        'learning_rate': 0.05, 'num_leaves': 63, 'min_data_in_leaf': 50,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'verbosity': -1, 'seed': 0,
    }
    print("\n=== iteration H: LightGBM LambdaRank ===")
    t0 = time.time()
    bst = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    print(f"  trained {bst.best_iteration} rounds in {time.time()-t0:.1f}s")

    scores_va = bst.predict(Xva, num_iteration=bst.best_iteration)
    metrics = evaluate(uva, yva, scores_va)
    print(f"  -> valid GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
          f"primary {metrics['primary']:.4f} | delta vs baseline "
          f"{metrics['primary']-BASELINE_VALID['primary']:+.4f} | delta vs FM-ensemble "
          f"{metrics['primary']-FM_ENSEMBLE_VALID_PRIMARY:+.4f}")
    importance = dict(sorted(zip(feature_names, bst.feature_importance('gain').tolist()),
                              key=lambda kv: -kv[1]))
    print("  feature importances (gain):", importance)

    np.save(os.path.join(a.out_dir, 'lgb_valid_scores.npy'), scores_va)
    np.save(os.path.join(a.out_dir, 'lgb_valid_users.npy'), np.array(uva))
    np.save(os.path.join(a.out_dir, 'lgb_valid_labels.npy'), yva)
    with open('logs/iteration_H_lgb.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'fm_ensemble_valid_primary': FM_ENSEMBLE_VALID_PRIMARY,
                    'lgb_valid': metrics, 'best_iteration': bst.best_iteration,
                    'feature_importance_gain': importance}, fh, indent=2, default=float)
    print(f"\nsaved scores to {a.out_dir}/, logged to logs/iteration_H_lgb.json")


if __name__ == '__main__':
    main()
