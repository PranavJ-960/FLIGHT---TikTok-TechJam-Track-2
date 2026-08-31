"""Iteration T: stacking retried with a RANKING-loss meta-learner instead of logistic
regression. The earlier attempt (stack_meta.py) failed badly because a global logistic
regression optimizes pointwise classification — nearly orthogonal to GAUC/nDCG@5, which
are within-user ranking metrics (raw model scores correlate ~0 with the label globally,
confirmed). A LightGBM lambdarank meta-learner, trained on the SAME out-of-fold base-model
predictions with the SAME per-user weighting and grouping as every other model in this
repo, respects the actual task structure instead. Requires experiments/stack_meta.py to
have already run and saved outputs/oof_{lgb,xgb,catboost}.npy + oof_users.npy +
oof_labels.npy. Train/valid only; standalone process.
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
from experiments.data_causal import encode_causal
from experiments.stack_meta import (build_lgb_matrix, build_cb_frame, train_lgb_once,
                                     train_xgb_once, train_catboost_once)

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LINEAR_BLEND_BEST = 0.6359


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)

    print("=== loading OOF predictions (from experiments/stack_meta.py) ===")
    oof_lgb = np.load(os.path.join(a.out_dir, 'oof_lgb.npy'))
    oof_xgb = np.load(os.path.join(a.out_dir, 'oof_xgb.npy'))
    oof_cat = np.load(os.path.join(a.out_dir, 'oof_catboost.npy'))
    oof_users = np.load(os.path.join(a.out_dir, 'oof_users.npy'), allow_pickle=True)
    oof_labels = np.load(os.path.join(a.out_dir, 'oof_labels.npy'))
    print(f"  {len(oof_labels)} OOF train rows")

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    meta_X = np.column_stack([norm(oof_lgb), norm(oof_xgb), norm(oof_cat)]).astype(np.float32)
    meta_feature_names = ['lgb_score', 'xgb_score', 'cat_score']

    order = np.argsort(oof_users, kind='stable')
    users_s = oof_users[order]
    X_s, y_s = meta_X[order], oof_labels[order]
    _, group = np.unique(users_s, return_counts=True)
    group_map = dict(zip(*np.unique(users_s, return_counts=True)))
    w = np.array([1.0 / group_map[u] for u in users_s], dtype=np.float32)

    print("=== fitting LightGBM lambdarank meta-learner on OOF predictions ===")
    # small model on purpose: only 3 input features, deliberately shallow/regularized to
    # avoid the meta-learner overfitting to OOF-specific noise (the user's overfitting
    # concern) — this is a combiner, not a full model, so it should stay simple.
    dtrain = lgb.Dataset(X_s, label=y_s, weight=w, group=group, feature_name=meta_feature_names)
    params = {'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [5], 'verbosity': -1,
              'seed': a.seed, 'learning_rate': 0.05, 'num_leaves': 7, 'max_depth': 3,
              'min_data_in_leaf': 500, 'feature_fraction': 1.0, 'bagging_fraction': 0.8,
              'lambda_l2': 5.0}
    meta = lgb.train(params, dtrain, num_boost_round=100)
    imp = dict(zip(meta_feature_names, meta.feature_importance('gain').tolist()))
    print(f"  meta-learner feature importances: {imp}")

    print("\n=== training each base model on FULL train, predicting valid (for final combination) ===")
    enc, feature_names = encode_causal(a.data_dir)
    Xtr_lgb, ytr, utr = build_lgb_matrix(enc, 'train')
    Xva_lgb, yva, uva = build_lgb_matrix(enc, 'valid')
    from experiments.data_causal import CAT_FIELDS, NUMERIC_FIELDS
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]
    Ctr, _, cutr = build_cb_frame(enc, 'train')
    Cva, _, cuva = build_cb_frame(enc, 'valid')
    cat_cols = ['dow'] + CAT_FIELDS
    assert utr == cutr and uva == cuva

    group_map_full = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    utr_s = np.array(utr)[order_tr]
    uva_s = np.array(uva)[order_va]
    _, group_tr_full = np.unique(utr_s, return_counts=True)
    _, group_va_full = np.unique(uva_s, return_counts=True)
    w_tr_full = np.array([1.0 / group_map_full[u] for u in utr_s], dtype=np.float32)

    t0 = time.time()
    valid_pred = {}
    valid_pred['lgb'] = train_lgb_once(Xtr_lgb[order_tr], ytr[order_tr], w_tr_full, group_tr_full,
                                        Xva_lgb[order_va], yva[order_va], group_va_full,
                                        feature_names, cat_idx, Xva_lgb)
    gw_full = (1.0 / group_tr_full).astype(np.float32)
    valid_pred['xgb'] = train_xgb_once(Xtr_lgb[order_tr], ytr[order_tr], gw_full, group_tr_full,
                                        Xva_lgb[order_va], yva[order_va], group_va_full,
                                        feature_names, Xva_lgb)
    valid_pred['catboost'] = train_catboost_once(Ctr.iloc[order_tr], ytr[order_tr], w_tr_full,
                                                  utr_s, Cva.iloc[order_va], yva[order_va],
                                                  uva_s, cat_cols, Cva)
    print(f"  base models trained in {time.time()-t0:.1f}s")
    for name in ('lgb', 'xgb', 'catboost'):
        m = evaluate(uva, yva, valid_pred[name])
        print(f"  {name}: valid primary={m['primary']:.4f}")

    meta_X_valid = np.column_stack([norm(valid_pred['lgb']), norm(valid_pred['xgb']),
                                     norm(valid_pred['catboost'])]).astype(np.float32)
    stacked_scores = meta.predict(meta_X_valid)
    stacked_metrics = evaluate(uva, yva, stacked_scores)
    print(f"\n=== GBT-stacked valid: GAUC {stacked_metrics['GAUC']:.4f} "
          f"nDCG@5 {stacked_metrics['nDCG@5']:.4f} primary {stacked_metrics['primary']:.4f} ===")
    print(f"delta vs baseline: {stacked_metrics['primary']-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs linear blend best ({LINEAR_BLEND_BEST}): "
          f"{stacked_metrics['primary']-LINEAR_BLEND_BEST:+.4f}")

    with open('logs/iteration_T_stack_meta_gbt.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'linear_blend_best': LINEAR_BLEND_BEST,
                    'meta_importance': imp, 'base_valid': {n: evaluate(uva, yva, valid_pred[n]) for n in valid_pred},
                    'stacked_valid': stacked_metrics}, fh, indent=2, default=float)
    print("logged to logs/iteration_T_stack_meta_gbt.json")


if __name__ == '__main__':
    main()
