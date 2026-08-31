"""1K bonus benchmark, round 2: (1) actually re-tune LightGBM for 1K instead of just
reusing Pure's config verbatim (Pure's config underperformed baseline last round —
0.6298 vs 0.6451 — plausibly because 10x more data + oversized-group chunking calls for
different num_leaves/min_data_in_leaf than Pure's much smaller, unchunked groups), (2) add
CatBoost to the 1K blend (previously skipped for cost reasons), (3) rerun XGBoost (already
strong at 0.6757) for a consistent 3-way blend. Reuses Pure's tuned CatBoost/XGBoost configs
verbatim (not re-tuned — still pragmatic for a bonus benchmark) but LightGBM gets a real
random search this time. Train/valid only, test never touched.
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

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS, NUMERIC_FIELDS

BASELINE_VALID = {'GAUC': 0.6749, 'nDCG@5': 0.6153, 'primary': 0.6451}
XGB_PARAMS = {'eta': 0.15, 'max_depth': 4, 'min_child_weight': 10, 'subsample': 1.0,
              'colsample_bytree': 1.0, 'lambda': 0.0, 'alpha': 1.0, 'gamma': 1.0}
CATBOOST_PARAMS = {'loss_function': 'YetiRank', 'eval_metric': 'NDCG:top=5',
                    'learning_rate': 0.15, 'depth': 8, 'l2_leaf_reg': 3.0,
                    'iterations': 1000, 'verbose': False, 'early_stopping_rounds': 30,
                    'thread_count': -1, 'subsample': 1.0, 'rsm': 1.0, 'random_strength': 1.0}
PREV_RESULT = {'LightGBM': 0.6298, 'XGBoost': 0.6757, 'blend': 0.6757}


def chunk_groups(sizes, max_size=10000):
    """LightGBM's lambdarank hard-rejects any single query group over 10000 rows."""
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


def build_frame(enc, split):
    d = enc[split]
    df = pd.DataFrame(d['num'], columns=NUMERIC_FIELDS)
    df['dow'] = d['dow'].astype(str)
    for i, c in enumerate(CAT_FIELDS):
        df[c] = d['cat'][:, i].astype(str)
    return df, d['y'], d['users']


def sample_lgb_params(rng):
    # Biased toward simpler trees than Pure's winning config (num_leaves=255, depth=5)
    # since 1K's oversized groups get chunked into pieces that are individually smaller/
    # noisier than Pure's native groups -- a plausible reason Pure's deep-tree config
    # underperformed baseline on 1K.
    return {
        'learning_rate': float(rng.choice([0.03, 0.05, 0.08, 0.12, 0.16])),
        'num_leaves': int(rng.choice([15, 31, 63, 127])),
        'min_data_in_leaf': int(rng.choice([50, 100, 200, 500, 1000])),
        'feature_fraction': float(rng.choice([0.5, 0.6, 0.7, 0.8, 1.0])),
        'bagging_fraction': float(rng.choice([0.6, 0.7, 0.8, 1.0])),
        'lambda_l1': float(rng.choice([0.0, 0.1, 1.0])),
        'lambda_l2': float(rng.choice([0.0, 1.0, 5.0, 10.0])),
        'max_depth': int(rng.choice([-1, 4, 5, 6, 8])),
        'min_gain_to_split': float(rng.choice([0.0, 0.01, 0.1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-1K/data_pure_names')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lgb_trials', type=int, default=10)
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

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

    print(f"\n=== LightGBM random search, {a.lgb_trials} trials (chunked groups) ===")
    group_tr_lgb = chunk_groups(group_tr)
    group_va_lgb = chunk_groups(group_va)
    print(f"  train {len(group_tr)}->{len(group_tr_lgb)} groups, valid {len(group_va)}->{len(group_va_lgb)} groups")
    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, weight=w_tr, group=group_tr_lgb, feature_name=feature_names,
                          categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va_lgb, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)

    lgb_trials = []
    best_lgb = None
    for t in range(a.lgb_trials):
        params = sample_lgb_params(rng)
        run_params = dict(params, objective='lambdarank', metric='ndcg', eval_at=[5],
                           verbosity=-1, seed=a.seed, feature_pre_filter=False)
        t0 = time.time()
        bst = lgb.train(run_params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        scores_va = bst.predict(Xva, num_iteration=bst.best_iteration)
        m = evaluate(uva, yva, scores_va)
        dt = time.time() - t0
        lgb_trials.append({'params': params, 'best_iteration': bst.best_iteration, 'valid': m, 'seconds': dt})
        flag = ""
        if best_lgb is None or m['primary'] > best_lgb['valid']['primary']:
            best_lgb = lgb_trials[-1]
            lgb_scores = scores_va
            flag = "  <- new best"
        print(f"  [{t+1:2d}/{a.lgb_trials}] primary={m['primary']:.4f} rounds={bst.best_iteration:3d} "
              f"({dt:.1f}s) {params}{flag}")
    m_lgb = best_lgb['valid']
    print(f"  best LightGBM: primary={m_lgb['primary']:.4f} (prev run: {PREV_RESULT['LightGBM']:.4f})")

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

    print("\n=== CatBoost (proven Pure config + per-row group_weight) ===")
    t0 = time.time()
    Xtr_df, _, _ = build_frame(enc, 'train')
    Xva_df, _, _ = build_frame(enc, 'valid')
    Xtr_df_s = Xtr_df.iloc[order_tr]
    Xva_df_s = Xva_df.iloc[order_va]
    cat_cols = ['dow'] + CAT_FIELDS
    train_pool = Pool(Xtr_df_s, label=ytr_s, group_id=utr_s, group_weight=w_tr, cat_features=cat_cols)
    valid_pool = Pool(Xva_df_s, label=yva_s, group_id=uva_s, cat_features=cat_cols)
    cb_params = dict(CATBOOST_PARAMS, random_seed=a.seed)
    model_cb = CatBoost(cb_params)
    model_cb.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    cb_valid_full = Pool(Xva_df, cat_features=cat_cols)
    cb_scores = model_cb.predict(cb_valid_full)
    m_cb = evaluate(uva, yva, cb_scores)
    print(f"  rounds={model_cb.get_best_iteration()} ({time.time()-t0:.1f}s) valid primary={m_cb['primary']:.4f}")

    print("\n=== 3-way blend ===")
    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    lgb_n, xgb_n, cb_n = norm(lgb_scores), norm(xgb_scores), norm(cb_scores)
    best = (-1, None)
    for w_l in np.arange(0.0, 1.01, 0.1):
        for w_x in np.arange(0.0, 1.01 - w_l, 0.1):
            w_c = 1.0 - w_l - w_x
            if w_c < -1e-9:
                continue
            blended = w_l * lgb_n + w_x * xgb_n + w_c * cb_n
            m = evaluate(uva, yva, blended)
            if m['primary'] > best[0]:
                best = (m['primary'], (round(float(w_l), 2), round(float(w_x), 2), round(float(w_c), 2)), m)
    best_primary, best_w, best_m = best
    print(f"  best blend: lgb={best_w[0]} xgb={best_w[1]} cat={best_w[2]} primary={best_primary:.4f}")

    print(f"\n{'model':20s} {'valid primary':>14s} {'delta vs 1K baseline':>21s}")
    print(f"{'baseline (FM)':20s} {BASELINE_VALID['primary']:14.4f} {'—':>21s}")
    print(f"{'LightGBM (tuned)':20s} {m_lgb['primary']:14.4f} {m_lgb['primary']-BASELINE_VALID['primary']:+21.4f}")
    print(f"{'XGBoost':20s} {m_xgb['primary']:14.4f} {m_xgb['primary']-BASELINE_VALID['primary']:+21.4f}")
    print(f"{'CatBoost':20s} {m_cb['primary']:14.4f} {m_cb['primary']-BASELINE_VALID['primary']:+21.4f}")
    print(f"{'3-way blend':20s} {best_primary:14.4f} {best_primary-BASELINE_VALID['primary']:+21.4f}")

    with open('logs/iteration_V_kuairand1k_v2.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'prev_result': PREV_RESULT,
                    'lgb_trials': lgb_trials, 'lgb_best': best_lgb, 'xgb_valid': m_xgb,
                    'catboost_valid': m_cb, 'catboost_params': cb_params,
                    'best_blend_weights_lgb_xgb_cat': best_w, 'best_blend_valid': best_m},
                  fh, indent=2, default=float)
    print("\nlogged to logs/iteration_V_kuairand1k_v2.json")


if __name__ == '__main__':
    main()
