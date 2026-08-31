"""ONE-TIME hidden-test scoring step for KuaiRand-Pure — run only when explicitly
requested by name, never invoked by any other script in this repo. Retrains the
exact winning configuration found by the compliant autonomous orchestrator run
(logs/orchestrator_run.json / logs/orchestrator_report.md: 14 iterations, 504.8s,
converged, best valid primary 0.6349 via a blend at iteration 10) — LightGBM
(iteration 8's params), XGBoost (iteration 4's params), CatBoost (iteration 3's
params), blended lgb=0.2/xgb=0.5/catboost=0.3 (iteration 10's weights) — on train,
then scores the test split exactly once: to write the submission CSV and report
the test-set delta over baseline. Never used to pick a model, tune a
hyperparameter, or influence anything upstream — everything upstream of this
script (feature set, hyperparameters, blend weights) was already decided on
train/valid alone before this file is ever run. See [[feedback-test-set-isolation]].
"""
import argparse
import csv
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

from starter_kit.data import load as official_load
from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS, NUMERIC_FIELDS

BASELINE_TEST = {'GAUC': 0.6610, 'nDCG@5': 0.5282, 'primary': 0.5946}   # starter_kit/baseline_scores.json, fm_official.test
BASELINE_VALID = {'GAUC': 0.6674, 'nDCG@5': 0.5357, 'primary': 0.6016}  # fm_official.valid
ORCHESTRATOR_VALID_PRIMARY = 0.6349   # logs/orchestrator_report.md, iteration 10

LGB_PARAMS = {'learning_rate': 0.12, 'num_leaves': 127, 'min_data_in_leaf': 100,
              'feature_fraction': 0.7, 'bagging_fraction': 0.9, 'lambda_l1': 0.1,
              'lambda_l2': 5.0, 'max_depth': 10, 'min_gain_to_split': 0.1}          # orchestrator iter 8
XGB_PARAMS = {'eta': 0.15, 'max_depth': 4, 'min_child_weight': 20, 'subsample': 1.0,
              'colsample_bytree': 0.5, 'lambda': 0.1, 'alpha': 0.1, 'gamma': 1.0}    # orchestrator iter 4
CB_PARAMS = {'learning_rate': 0.15, 'depth': 7, 'l2_leaf_reg': 10.0,
             'subsample': 0.8, 'rsm': 1.0, 'random_strength': 0.0}                  # orchestrator iter 3
BLEND_WEIGHTS = {'lgb': 0.2, 'xgb': 0.5, 'catboost': 0.3}                            # orchestrator iter 10


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def build_frame(enc, split):
    d = enc[split]
    df = pd.DataFrame(d['num'], columns=NUMERIC_FIELDS)
    df['dow'] = d['dow'].astype(str)
    for i, c in enumerate(CAT_FIELDS):
        df[c] = d['cat'][:, i].astype(str)
    return df, d['y'], d['users']


def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_csv', default='submission_pure.csv')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)

    print("=== [ONE-TIME] building causal features (train / valid / test) ===")
    t0 = time.time()
    enc, feature_names = encode_causal(a.data_dir)
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]
    print(f"  ready in {time.time()-t0:.1f}s")

    Xtr, ytr, utr = build_matrix(enc, 'train')
    Xva, yva, uva = build_matrix(enc, 'valid')
    Xte, yte, ute = build_matrix(enc, 'test')

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    utr_s = np.array([utr[i] for i in order_tr])
    uva_s = np.array([uva[i] for i in order_va])
    group_tr = np.unique(utr_s, return_counts=True)[1]
    group_va = np.unique(uva_s, return_counts=True)[1]
    group_map = dict(zip(*np.unique(utr_s, return_counts=True)))
    row_w_tr = np.array([1.0 / group_map[u] for u in utr_s], dtype=np.float32)
    group_w_tr = (1.0 / group_tr).astype(np.float32)

    # ---------------- LightGBM ----------------
    print("\n=== training LightGBM (orchestrator iteration 8's config) ===")
    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, weight=row_w_tr, group=group_tr,
                          feature_name=feature_names, categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)
    run_params = dict(LGB_PARAMS, objective='lambdarank', metric='ndcg', eval_at=[5],
                       verbosity=-1, seed=a.seed, feature_pre_filter=False)
    t0 = time.time()
    lgb_model = lgb.train(run_params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                           callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    print(f"  trained {lgb_model.best_iteration} rounds in {time.time()-t0:.1f}s")
    lgb_va = lgb_model.predict(Xva, num_iteration=lgb_model.best_iteration)
    lgb_te = lgb_model.predict(Xte, num_iteration=lgb_model.best_iteration)

    # ---------------- XGBoost ----------------
    print("\n=== training XGBoost (orchestrator iteration 4's config) ===")
    xtrain = xgb.DMatrix(Xtr_s, label=ytr_s, feature_names=feature_names)
    xtrain.set_weight(group_w_tr)
    xtrain.set_group(group_tr)
    xvalid = xgb.DMatrix(Xva_s, label=yva_s, feature_names=feature_names)
    xvalid.set_group(group_va)
    xvalid_full = xgb.DMatrix(Xva, feature_names=feature_names)
    xtest_full = xgb.DMatrix(Xte, feature_names=feature_names)
    run_params_x = dict(XGB_PARAMS, objective='rank:ndcg', eval_metric='ndcg@5', seed=a.seed, verbosity=0)
    t0 = time.time()
    xgb_model = xgb.train(run_params_x, xtrain, num_boost_round=1000, evals=[(xvalid, 'valid')],
                           early_stopping_rounds=30, verbose_eval=False)
    print(f"  trained {xgb_model.best_iteration} rounds in {time.time()-t0:.1f}s")
    xgb_va = xgb_model.predict(xvalid_full, iteration_range=(0, xgb_model.best_iteration + 1))
    xgb_te = xgb_model.predict(xtest_full, iteration_range=(0, xgb_model.best_iteration + 1))

    # ---------------- CatBoost ----------------
    print("\n=== training CatBoost (orchestrator iteration 3's config) ===")
    Ctr, cytr, cutr = build_frame(enc, 'train')
    Cva, cyva, cuva = build_frame(enc, 'valid')
    Cte, cyte, cute = build_frame(enc, 'test')
    cat_cols = ['dow'] + CAT_FIELDS
    order_ctr = np.argsort(cutr, kind='stable')
    order_cva = np.argsort(cuva, kind='stable')
    group_ctr = np.array([cutr[i] for i in order_ctr])
    group_cva = np.array([cuva[i] for i in order_cva])
    row_w_ctr = np.array([1.0 / group_map[u] for u in group_ctr], dtype=np.float32)
    cb_train = Pool(Ctr.iloc[order_ctr], label=cytr[order_ctr], group_id=group_ctr,
                     group_weight=row_w_ctr, cat_features=cat_cols)
    cb_valid = Pool(Cva.iloc[order_cva], label=cyva[order_cva], group_id=group_cva, cat_features=cat_cols)
    cb_valid_full = Pool(Cva, cat_features=cat_cols)
    cb_test_full = Pool(Cte, cat_features=cat_cols)
    run_params_c = dict(CB_PARAMS, loss_function='YetiRank', eval_metric='NDCG:top=5',
                         iterations=400, random_seed=a.seed, verbose=False,
                         early_stopping_rounds=20, thread_count=-1)
    t0 = time.time()
    cb_model = CatBoost(run_params_c)
    cb_model.fit(cb_train, eval_set=cb_valid, use_best_model=True)
    print(f"  trained {cb_model.get_best_iteration()} rounds in {time.time()-t0:.1f}s")
    cb_va = cb_model.predict(cb_valid_full)
    cb_te = cb_model.predict(cb_test_full)

    # ---------------- blend (orchestrator iteration 10's weights) ----------------
    print(f"\n=== blending {BLEND_WEIGHTS} ===")
    blend_va = (BLEND_WEIGHTS['lgb'] * norm(lgb_va) + BLEND_WEIGHTS['xgb'] * norm(xgb_va) +
                BLEND_WEIGHTS['catboost'] * norm(cb_va))
    blend_te = (BLEND_WEIGHTS['lgb'] * norm(lgb_te) + BLEND_WEIGHTS['xgb'] * norm(xgb_te) +
                BLEND_WEIGHTS['catboost'] * norm(cb_te))

    valid_metrics = evaluate(uva, yva, blend_va)
    test_metrics = evaluate(ute, yte, blend_te)   # the one-time, explicitly-requested scoring
    print(f"\nvalid: GAUC {valid_metrics['GAUC']:.4f} nDCG@5 {valid_metrics['nDCG@5']:.4f} "
          f"primary {valid_metrics['primary']:.4f} (orchestrator run reported {ORCHESTRATOR_VALID_PRIMARY:.4f})")
    print(f"test:  GAUC {test_metrics['GAUC']:.4f} nDCG@5 {test_metrics['nDCG@5']:.4f} "
          f"primary {test_metrics['primary']:.4f}")
    print(f"test delta vs official baseline: GAUC {test_metrics['GAUC']-BASELINE_TEST['GAUC']:+.4f} "
          f"nDCG@5 {test_metrics['nDCG@5']-BASELINE_TEST['nDCG@5']:+.4f} "
          f"primary {test_metrics['primary']-BASELINE_TEST['primary']:+.4f}")

    # ---------------- write submission.csv (row order per the OFFICIAL loader) ----------------
    official_splits = official_load(a.data_dir)
    test_rows = official_splits['test']
    assert len(test_rows) == len(ute), f"row count mismatch: official={len(test_rows)} causal={len(ute)}"
    assert [x[1] for x in test_rows] == ute, "row order mismatch: official loader vs causal-feature loader (users)"
    print(f"\nrow-order alignment check passed: {len(test_rows):,} test rows, user_id sequences match")

    with open(a.out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (x, s) in enumerate(zip(test_rows, blend_te)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    print(f"wrote {a.out_csv} ({len(test_rows):,} rows)")

    with open('logs/hidden_test_score.json', 'w') as fh:
        json.dump({
            'baseline_valid': BASELINE_VALID, 'baseline_test': BASELINE_TEST,
            'orchestrator_valid_primary': ORCHESTRATOR_VALID_PRIMARY,
            'lgb_params': LGB_PARAMS, 'xgb_params': XGB_PARAMS, 'catboost_params': CB_PARAMS,
            'blend_weights': BLEND_WEIGHTS,
            'valid_metrics': valid_metrics, 'test_metrics': test_metrics,
            'test_delta_vs_baseline': {k: test_metrics[k] - BASELINE_TEST[k] for k in ('GAUC', 'nDCG@5', 'primary')},
        }, fh, indent=2, default=float)
    print("logged to logs/hidden_test_score.json")


if __name__ == '__main__':
    main()
