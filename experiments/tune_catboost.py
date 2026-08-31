"""Tune CatBoost (YetiRank) on the causal features. CatBoost is much slower than
LightGBM/XGBoost at this data size (~100s/run) so this uses a smaller trial budget.
Train/valid only.
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
CAT_DEFAULT = 0.6263


def build_frame(enc, split):
    d = enc[split]
    df = pd.DataFrame(d['num'], columns=NUMERIC_FIELDS)
    df['dow'] = d['dow'].astype(str)
    for i, c in enumerate(CAT_FIELDS):
        df[c] = d['cat'][:, i].astype(str)
    return df, d['y'], d['users']


def sample_params(rng):
    return {
        'learning_rate': float(rng.choice([0.03, 0.05, 0.08, 0.1, 0.15])),
        'depth': int(rng.choice([4, 5, 6, 7, 8])),
        'l2_leaf_reg': float(rng.choice([1.0, 3.0, 5.0, 10.0])),
        'subsample': float(rng.choice([0.6, 0.8, 1.0])),
        'rsm': float(rng.choice([0.6, 0.8, 1.0])),
        'random_strength': float(rng.choice([0.0, 1.0, 5.0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--n_trials', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

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

    # per-group weight (1/group size) — CatBoost's pairwise losses (YetiRank) reject
    # per-row `weight` ("Pairwise losses don't support object weights"); `group_weight`
    # is the supported equivalent. See experiments/tune_lgb.py's comment for the intent.
    group_map = dict(zip(*np.unique(group_tr, return_counts=True)))
    gw_tr = np.array([1.0 / group_map[u] for u in group_tr], dtype=np.float32)

    train_pool = Pool(Xtr_s, label=ytr_s, group_id=group_tr, group_weight=gw_tr, cat_features=cat_cols)
    valid_pool = Pool(Xva_s, label=yva_s, group_id=group_va, cat_features=cat_cols)
    valid_pool_full = Pool(Xva, cat_features=cat_cols)

    rng = np.random.default_rng(a.seed)
    print(f"=== tuning CatBoost, {a.n_trials} trials (slow, ~1-2min/trial) ===")
    best, best_scores = None, None
    for t in range(a.n_trials):
        params = sample_params(rng)
        run_params = dict(params, loss_function='YetiRank', eval_metric='NDCG:top=5',
                           iterations=600, random_seed=a.seed, verbose=False,
                           early_stopping_rounds=25, thread_count=-1)
        t0 = time.time()
        model = CatBoost(run_params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        scores = model.predict(valid_pool_full)
        m = evaluate(uva, yva, scores)
        dt = time.time() - t0
        flag = ""
        if best is None or m['primary'] > best['valid']['primary']:
            best = {'params': params, 'valid': m, 'best_iteration': model.get_best_iteration()}
            best_scores = scores
            flag = "  <- new best"
        print(f"  [{t+1:2d}/{a.n_trials}] primary={m['primary']:.4f} rounds={model.get_best_iteration():3d} "
              f"({dt:.1f}s) {params}{flag}", flush=True)

    print(f"\nbest: primary={best['valid']['primary']:.4f} "
          f"(delta vs baseline {best['valid']['primary']-BASELINE_VALID['primary']:+.4f}, "
          f"delta vs untuned {best['valid']['primary']-CAT_DEFAULT:+.4f})")
    print(f"best params: {best['params']}")

    np.save(os.path.join(a.out_dir, 'catboost_valid_scores.npy'), best_scores)
    np.save(os.path.join(a.out_dir, 'catboost_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'catboost_valid_labels.npy'), yva)
    with open('logs/iteration_Q_catboost_tune.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'catboost_untuned': CAT_DEFAULT, 'best': best},
                   fh, indent=2, default=float)
    print("logged to logs/iteration_Q_catboost_tune.json (and overwrote outputs/catboost_valid_scores.npy)")


if __name__ == '__main__':
    main()
