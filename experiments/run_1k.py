"""Apply the proven Pure pipeline (causal features + per-user-weighted LightGBM/XGBoost,
blended) to KuaiRand-1K for bonus points. Reuses Pure's already-tuned best hyperparameters
rather than re-tuning from scratch — pragmatic for a bonus benchmark, not the main
optimization target. Data accessed via data/KuaiRand-1K/data_pure_names (symlinks to the
1K files under starter_kit-compatible names) so starter_kit/ needs zero changes. Loads
causal features once and reuses in-memory across both models (1K's feature build is ~150s,
too expensive to redo per model). Train/valid only.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS

LGB_PARAMS = {'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 20,
              'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'lambda_l1': 0.1,
              'lambda_l2': 1.0, 'max_depth': 5, 'min_gain_to_split': 0.01}
XGB_PARAMS = {'eta': 0.15, 'max_depth': 4, 'min_child_weight': 10, 'subsample': 1.0,
              'colsample_bytree': 1.0, 'lambda': 0.0, 'alpha': 1.0, 'gamma': 1.0}
BASELINE_VALID = {'GAUC': 0.6749, 'nDCG@5': 0.6153, 'primary': 0.6451}   # reproduced FM, seed0


def chunk_groups(sizes, max_size=10000):
    """LightGBM's lambdarank rejects any single query group over 10000 rows — fine on
    Pure (max ~a few hundred impressions/user) but 1K covers FULL logs per user, so some
    power users exceed it by a lot (up to ~49k rows). Split oversized groups into several
    consecutive sub-groups; row order is unchanged, only how they're partitioned for the
    ranking loss. Only affects LightGBM's internal training signal for those users, not
    the final reported metric — evaluate() always uses the true, unchunked per-user
    grouping. XGBoost has no such limit and keeps using the true groups throughout."""
    new_sizes = []
    for s in sizes:
        if s <= max_size:
            new_sizes.append(s)
        else:
            n_chunks = -(-int(s) // max_size)
            base, rem = divmod(int(s), n_chunks)
            new_sizes.extend(base + (1 if i < rem else 0) for i in range(n_chunks))
    return np.array(new_sizes, dtype=np.int32)


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-1K/data_pure_names')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    print("=== building causal features (1K scale, ~150s) ===")
    t0 = time.time()
    enc, feature_names = encode_causal(a.data_dir)
    print(f"  ready in {time.time()-t0:.1f}s, {len(feature_names)} features")

    Xtr, ytr, utr = build_matrix(enc, 'train')
    Xva, yva, uva = build_matrix(enc, 'valid')
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    utr_s = np.array([utr[i] for i in order_tr])
    uva_s = np.array([uva[i] for i in order_va])
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    _, group_tr = np.unique(utr_s, return_counts=True)
    _, group_va = np.unique(uva_s, return_counts=True)
    group_map = dict(zip(*np.unique(utr_s, return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr_s], dtype=np.float32)

    print("\n=== LightGBM (proven Pure config + per-user weighting) ===")
    t0 = time.time()
    group_tr_lgb = chunk_groups(group_tr)
    group_va_lgb = chunk_groups(group_va)
    print(f"  chunked oversized (>10000-row) query groups for LightGBM: "
          f"train {len(group_tr)}->{len(group_tr_lgb)} groups, valid {len(group_va)}->{len(group_va_lgb)} groups")
    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, weight=w_tr, group=group_tr_lgb, feature_name=feature_names,
                          categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va_lgb, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)
    params = dict(LGB_PARAMS, objective='lambdarank', metric='ndcg', eval_at=[5],
                  verbosity=-1, seed=a.seed, feature_pre_filter=False)
    bst_lgb = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    lgb_scores = bst_lgb.predict(Xva, num_iteration=bst_lgb.best_iteration)
    m_lgb = evaluate(uva, yva, lgb_scores)
    print(f"  rounds={bst_lgb.best_iteration} ({time.time()-t0:.1f}s) valid primary={m_lgb['primary']:.4f}")

    print("\n=== XGBoost (proven Pure config + per-group weighting) ===")
    t0 = time.time()
    group_weight_tr = (1.0 / group_tr).astype(np.float32)
    dtr_xgb = xgb.DMatrix(Xtr_s, label=ytr_s, feature_names=feature_names)
    dtr_xgb.set_weight(group_weight_tr)
    dtr_xgb.set_group(group_tr)
    dva_xgb = xgb.DMatrix(Xva_s, label=yva_s, feature_names=feature_names)
    dva_xgb.set_group(group_va)
    dva_full = xgb.DMatrix(Xva, feature_names=feature_names)
    params_x = dict(XGB_PARAMS, objective='rank:ndcg', eval_metric='ndcg@5', seed=a.seed, verbosity=0)
    bst_xgb = xgb.train(params_x, dtr_xgb, num_boost_round=1000, evals=[(dva_xgb, 'valid')],
                         early_stopping_rounds=30, verbose_eval=False)
    xgb_scores = bst_xgb.predict(dva_full, iteration_range=(0, bst_xgb.best_iteration + 1))
    m_xgb = evaluate(uva, yva, xgb_scores)
    print(f"  rounds={bst_xgb.best_iteration} ({time.time()-t0:.1f}s) valid primary={m_xgb['primary']:.4f}")

    print("\n=== blend ===")
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    lgb_n, xgb_n = norm(lgb_scores), norm(xgb_scores)
    best = (-1, None)
    for w in np.arange(0.0, 1.01, 0.1):
        blended = w * lgb_n + (1 - w) * xgb_n
        m = evaluate(uva, yva, blended)
        if m['primary'] > best[0]:
            best = (m['primary'], (round(float(w), 2), m))
    best_primary, (best_w, best_m) = best
    print(f"  best blend: w_lgb={best_w} primary={best_primary:.4f}")

    print(f"\n{'model':20s} {'valid primary':>14s} {'delta vs 1K baseline':>21s}")
    print(f"{'baseline (FM)':20s} {BASELINE_VALID['primary']:14.4f} {'—':>21s}")
    print(f"{'LightGBM':20s} {m_lgb['primary']:14.4f} {m_lgb['primary']-BASELINE_VALID['primary']:+21.4f}")
    print(f"{'XGBoost':20s} {m_xgb['primary']:14.4f} {m_xgb['primary']-BASELINE_VALID['primary']:+21.4f}")
    print(f"{'blend':20s} {best_primary:14.4f} {best_primary-BASELINE_VALID['primary']:+21.4f}")

    with open('logs/iteration_U_kuairand1k.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'lgb_valid': m_lgb, 'xgb_valid': m_xgb,
                    'best_blend_weight_lgb': best_w, 'best_blend_valid': best_m}, fh, indent=2, default=float)
    print("\nlogged to logs/iteration_U_kuairand1k.json")


if __name__ == '__main__':
    main()
