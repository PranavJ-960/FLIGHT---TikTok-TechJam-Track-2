"""Iteration J: hyperparameter search for LightGBM on the (now richer) causal
features. Random search over a bounded grid, scored on valid via starter_kit's
evaluate() (not lightgbm's own internal metric, kept only for early stopping) —
train/valid only, test never unpacked/evaluated. Standalone process, no torch
import (see experiments/train_lgb.py's docstring on the torch/lightgbm deadlock).
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
LGB_DEFAULT_PRIMARY = 0.6137   # richer features, un-tuned params (iteration H)


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def sample_params(rng):
    return {
        'learning_rate': float(rng.choice([0.03, 0.05, 0.08, 0.12, 0.16, 0.2])),
        'num_leaves': int(rng.choice([31, 63, 127, 255, 400])),
        'min_data_in_leaf': int(rng.choice([10, 20, 50, 100, 150])),
        'feature_fraction': float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
        'bagging_fraction': float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
        'lambda_l1': float(rng.choice([0.0, 0.1, 1.0])),
        'lambda_l2': float(rng.choice([0.0, 0.1, 1.0, 5.0, 10.0])),
        'max_depth': int(rng.choice([-1, 5, 6, 8, 10, 12])),
        'min_gain_to_split': float(rng.choice([0.0, 0.01, 0.1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--n_trials', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--with_relative', action='store_true',
                     help='add within-user relative/percentile-rank features')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("=== building causal features ===")
    t0 = time.time()
    enc, feature_names = encode_causal(a.data_dir, with_relative=a.with_relative)
    print(f"  built in {time.time()-t0:.1f}s")

    Xtr, ytr, utr = build_matrix(enc, 'train')
    Xva, yva, uva = build_matrix(enc, 'valid')
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    utr_s = np.array([utr[i] for i in order_tr])
    uva_s = np.array([uva[i] for i in order_va])
    _, group_tr = np.unique(utr_s, return_counts=True)
    _, group_va = np.unique(uva_s, return_counts=True)

    # per-user row weight (1/group size): GAUC/nDCG@5 average per USER, but pointwise/
    # pairwise loss otherwise weights each ROW equally, so a 200-impression user gets 200x
    # the training influence of a 1-impression user despite counting the same in the score.
    group_map = dict(zip(*np.unique(utr_s, return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr_s], dtype=np.float32)

    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, weight=w_tr, group=group_tr, feature_name=feature_names,
                          categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)

    rng = np.random.default_rng(a.seed)
    print(f"\n=== iteration J: random search, {a.n_trials} trials ===")
    trials = []
    best = None
    for t in range(a.n_trials):
        params = sample_params(rng)
        run_params = dict(params, objective='lambdarank', metric='ndcg', eval_at=[5],
                           verbosity=-1, seed=a.seed, feature_pre_filter=False)
        t0 = time.time()
        bst = lgb.train(run_params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        scores_va = bst.predict(Xva, num_iteration=bst.best_iteration)
        m = evaluate(uva, yva, scores_va)
        dt = time.time() - t0
        trials.append({'params': params, 'best_iteration': bst.best_iteration, 'valid': m, 'seconds': dt})
        flag = ""
        if best is None or m['primary'] > best['valid']['primary']:
            best = trials[-1]
            best_scores = scores_va
            flag = "  <- new best"
        print(f"  [{t+1:2d}/{a.n_trials}] primary={m['primary']:.4f} rounds={bst.best_iteration:3d} "
              f"({dt:.1f}s) {params}{flag}")

    print(f"\nbest trial: primary={best['valid']['primary']:.4f} "
          f"(delta vs baseline {best['valid']['primary']-BASELINE_VALID['primary']:+.4f}, "
          f"delta vs untuned-richer-features {best['valid']['primary']-LGB_DEFAULT_PRIMARY:+.4f})")
    print(f"best params: {best['params']}")

    np.save(os.path.join(a.out_dir, 'lgb_tuned_valid_scores.npy'), best_scores)
    np.save(os.path.join(a.out_dir, 'lgb_tuned_valid_users.npy'), np.array(uva))
    np.save(os.path.join(a.out_dir, 'lgb_tuned_valid_labels.npy'), yva)
    with open('logs/iteration_J_lgb_tune.json', 'w') as fh:
        json.dump({
            'baseline_valid': BASELINE_VALID,
            'lgb_untuned_richer_features_primary': LGB_DEFAULT_PRIMARY,
            'trials': trials,
            'best': best,
        }, fh, indent=2, default=float)
    print("logged to logs/iteration_J_lgb_tune.json")


if __name__ == '__main__':
    main()
