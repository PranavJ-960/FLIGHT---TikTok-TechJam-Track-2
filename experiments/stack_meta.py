"""Iteration R: proper out-of-fold stacking — a structurally different combination
method from the linear weight search in blend_multi.py. K-fold out-of-fold (OOF)
predictions from LightGBM/XGBoost/CatBoost (each at its already-tuned best
hyperparameters) are used to fit a meta-learner (logistic regression) that can learn
a genuinely non-linear/conditional combination, rather than one fixed weight per
model applied uniformly to every row. The meta-learner is then applied to each base
model's predictions on the true valid split (already-tuned single runs, same as
blend_multi.py uses). Train/valid only — test never touched. Standalone process
(no torch import).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool
from sklearn.linear_model import LogisticRegression

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS, NUMERIC_FIELDS

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LINEAR_BLEND_BEST = 0.6359

LGB_PARAMS = {'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 20,
              'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'lambda_l1': 0.1,
              'lambda_l2': 1.0, 'max_depth': 5, 'min_gain_to_split': 0.01}
XGB_PARAMS = {'eta': 0.15, 'max_depth': 4, 'min_child_weight': 10, 'subsample': 1.0,
              'colsample_bytree': 1.0, 'lambda': 0.0, 'alpha': 1.0, 'gamma': 1.0}
CB_PARAMS = {'learning_rate': 0.08, 'depth': 5, 'l2_leaf_reg': 1.0, 'subsample': 0.6,
             'rsm': 0.8, 'random_strength': 5.0}


def build_lgb_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def build_cb_frame(enc, split):
    d = enc[split]
    df = pd.DataFrame(d['num'], columns=NUMERIC_FIELDS)
    df['dow'] = d['dow'].astype(str)
    for i, c in enumerate(CAT_FIELDS):
        df[c] = d['cat'][:, i].astype(str)
    return df, d['y'], d['users']


def train_lgb_once(Xtr, ytr, wtr, group_tr, Xva, yva, group_va, feature_names, cat_idx, X_predict):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, group=group_tr, feature_name=feature_names,
                       categorical_feature=cat_idx)
    dva = lgb.Dataset(Xva, label=yva, group=group_va, feature_name=feature_names,
                       categorical_feature=cat_idx, reference=dtr)
    params = dict(LGB_PARAMS, objective='lambdarank', metric='ndcg', eval_at=[5],
                  verbosity=-1, seed=0, feature_pre_filter=False)
    bst = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return bst.predict(X_predict, num_iteration=bst.best_iteration)


def train_xgb_once(Xtr, ytr, group_w_tr, group_tr, Xva, yva, group_va, feature_names, X_predict):
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_names)
    dtr.set_weight(group_w_tr)
    dtr.set_group(group_tr)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feature_names)
    dva.set_group(group_va)
    params = dict(XGB_PARAMS, objective='rank:ndcg', eval_metric='ndcg@5', seed=0, verbosity=0)
    bst = xgb.train(params, dtr, num_boost_round=1000, evals=[(dva, 'valid')],
                     early_stopping_rounds=30, verbose_eval=False)
    dpred = xgb.DMatrix(X_predict, feature_names=feature_names)
    return bst.predict(dpred, iteration_range=(0, bst.best_iteration + 1))


def train_catboost_once(Xtr_df, ytr, row_w_tr, group_tr, Xva_df, yva, group_va, cat_cols, X_predict_df):
    tr_pool = Pool(Xtr_df, label=ytr, group_id=group_tr, group_weight=row_w_tr, cat_features=cat_cols)
    va_pool = Pool(Xva_df, label=yva, group_id=group_va, cat_features=cat_cols)
    params = dict(CB_PARAMS, loss_function='YetiRank', eval_metric='NDCG:top=5',
                  iterations=600, random_seed=0, verbose=False, early_stopping_rounds=25, thread_count=-1)
    model = CatBoost(params)
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    pred_pool = Pool(X_predict_df, cat_features=cat_cols)
    return model.predict(pred_pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--n_folds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print("=== building causal features ===")
    enc, feature_names = encode_causal(a.data_dir)
    Xtr_lgb, ytr, utr = build_lgb_matrix(enc, 'train')
    Xva_lgb, yva, uva = build_lgb_matrix(enc, 'valid')
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]
    Ctr, _, cutr = build_cb_frame(enc, 'train')
    Cva, _, cuva = build_cb_frame(enc, 'valid')
    cat_cols = ['dow'] + CAT_FIELDS
    assert utr == cutr and uva == cuva

    n = len(utr)
    fold_id = rng.integers(0, a.n_folds, size=n)
    group_map_full = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    w_full = np.array([1.0 / group_map_full[u] for u in utr], dtype=np.float32)

    oof = {'lgb': np.zeros(n, dtype=np.float32), 'xgb': np.zeros(n, dtype=np.float32),
           'catboost': np.zeros(n, dtype=np.float32)}

    print(f"=== generating {a.n_folds}-fold out-of-fold predictions ===")
    for k in range(a.n_folds):
        t0 = time.time()
        held = fold_id == k
        fit = ~held
        fit_users = np.array(utr)[fit]
        order = np.argsort(fit_users, kind='stable')

        # ---- lgb ----
        Xf, yf = Xtr_lgb[fit][order], ytr[fit][order]
        wf = w_full[fit][order]
        _, gtr = np.unique(fit_users[order], return_counts=True)
        held_users = np.array(utr)[held]
        order_h = np.argsort(held_users, kind='stable')
        Xh, yh = Xtr_lgb[held][order_h], ytr[held][order_h]
        _, ghe = np.unique(held_users[order_h], return_counts=True)
        pred = train_lgb_once(Xf, yf, wf, gtr, Xh, yh, ghe, feature_names, cat_idx, Xtr_lgb[held])
        oof['lgb'][held] = pred
        print(f"  fold {k}: lgb done ({time.time()-t0:.1f}s)", flush=True)

        # ---- xgb ----
        t0 = time.time()
        gw = (1.0 / gtr).astype(np.float32)
        pred = train_xgb_once(Xf, yf, gw, gtr, Xh, yh, ghe, feature_names, Xtr_lgb[held])
        oof['xgb'][held] = pred
        print(f"  fold {k}: xgb done ({time.time()-t0:.1f}s)", flush=True)

        # ---- catboost ----
        t0 = time.time()
        Cf = Ctr.iloc[fit].iloc[order]
        Ch = Ctr.iloc[held].iloc[order_h]
        row_w_f = wf   # per-row weight, matches CatBoost's group_weight convention
        pred = train_catboost_once(Cf, yf, row_w_f, fit_users[order], Ch, yh, held_users[order_h],
                                    cat_cols, Ctr[held])
        oof['catboost'][held] = pred
        print(f"  fold {k}: catboost done ({time.time()-t0:.1f}s)", flush=True)

    for name in ('lgb', 'xgb', 'catboost'):
        m = evaluate(utr, ytr, oof[name])   # sanity check only — OOF-on-train isn't comparable to valid numbers
        print(f"  OOF sanity {name}: primary={m['primary']:.4f}")

    os.makedirs('outputs', exist_ok=True)
    for name in ('lgb', 'xgb', 'catboost'):
        np.save(f"outputs/oof_{name}.npy", oof[name])
    np.save("outputs/oof_users.npy", np.array(utr, dtype=object))
    np.save("outputs/oof_labels.npy", ytr)
    print("  saved OOF arrays to outputs/oof_*.npy (reusable by other meta-learners)")

    print("\n=== fitting meta-learners (logistic regression, C sweep) on OOF predictions ===")
    # unregularized (or lightly regularized) logistic regression on 3 highly-correlated
    # OOF scores is unstable — huge, opposite-signed coefficients that fit train-OOF noise
    # rather than a real combination pattern (confirmed: C=1.0 gave primary 0.5642 on valid,
    # WORSE than baseline). Sweep regularization strength and pick by valid, same as every
    # other hyperparameter choice in this pipeline.
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    meta_X_train = np.column_stack([norm(oof['lgb']), norm(oof['xgb']), norm(oof['catboost'])])
    metas = {}
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        m = LogisticRegression(C=C)
        m.fit(meta_X_train, ytr)
        metas[C] = m
        print(f"  C={C:<6} coefficients: lgb={m.coef_[0][0]:.3f} xgb={m.coef_[0][1]:.3f} "
              f"catboost={m.coef_[0][2]:.3f} intercept={m.intercept_[0]:.3f}")

    print("\n=== training each base model on FULL train, predicting valid ===")
    group_map = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    utr_s = np.array(utr)[order_tr]
    uva_s = np.array(uva)[order_va]
    _, group_tr_full = np.unique(utr_s, return_counts=True)
    _, group_va_full = np.unique(uva_s, return_counts=True)
    w_tr_full = np.array([1.0 / group_map[u] for u in utr_s], dtype=np.float32)

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

    for name in ('lgb', 'xgb', 'catboost'):
        m = evaluate(uva, yva, valid_pred[name])
        print(f"  {name}: valid primary={m['primary']:.4f}")

    meta_X_valid = np.column_stack([norm(valid_pred['lgb']), norm(valid_pred['xgb']), norm(valid_pred['catboost'])])
    print("\n=== stacked (meta-learner) valid, by C ===")
    results_by_C = {}
    best_C, best_metrics = None, None
    for C, m in metas.items():
        stacked_scores = m.predict_proba(meta_X_valid)[:, 1]
        sm = evaluate(uva, yva, stacked_scores)
        results_by_C[C] = sm
        print(f"  C={C:<6} primary={sm['primary']:.4f} (delta vs baseline {sm['primary']-BASELINE_VALID['primary']:+.4f}, "
              f"delta vs linear blend {sm['primary']-LINEAR_BLEND_BEST:+.4f})")
        if best_metrics is None or sm['primary'] > best_metrics['primary']:
            best_C, best_metrics = C, sm

    print(f"\nbest: C={best_C} primary={best_metrics['primary']:.4f} "
          f"(delta vs baseline {best_metrics['primary']-BASELINE_VALID['primary']:+.4f}, "
          f"delta vs linear blend best ({LINEAR_BLEND_BEST}): {best_metrics['primary']-LINEAR_BLEND_BEST:+.4f})")

    best_meta = metas[best_C]
    with open('logs/iteration_R_stack_meta.json', 'w') as fh:
        json.dump({
            'baseline_valid': BASELINE_VALID, 'linear_blend_best': LINEAR_BLEND_BEST,
            'best_C': best_C,
            'best_meta_coefficients': {'lgb': float(best_meta.coef_[0][0]), 'xgb': float(best_meta.coef_[0][1]),
                                         'catboost': float(best_meta.coef_[0][2]), 'intercept': float(best_meta.intercept_[0])},
            'base_valid': {name: evaluate(uva, yva, valid_pred[name]) for name in valid_pred},
            'results_by_C': results_by_C,
            'best_stacked_valid': best_metrics,
        }, fh, indent=2, default=float)
    print("logged to logs/iteration_R_stack_meta.json")


if __name__ == '__main__':
    main()
