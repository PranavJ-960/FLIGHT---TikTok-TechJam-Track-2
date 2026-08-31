"""Censored watch-time regression (a lightweight, from-scratch take on the CWM idea
distinct from any FM/DCN/DeepFM variant here) — instead of ranking on the binary
long_view label directly, regress watch_ratio = play_time_ms / duration_ms and use
the predicted ratio as the ranking score. play_time_ms can exceed duration_ms in
KuaiRand (looped plays), which right-censors the *true* engagement at ratio=1: we
only know it's "at least fully watched," not how much more. A plain regression loss
on the raw (possibly-looped) ratio would chase noisy loop counts; plain clipping at
1.0 would throw away the "definitely fully watched" signal entirely and just
reproduce a fuzzy long_view. Instead this trains a one-sided (Tobit-style) squared
loss via a custom LightGBM objective: uncensored rows (ratio < 1) get ordinary
squared-error gradients; censored rows (ratio >= 1) only get gradient when the
prediction *undershoots* 1.0, none when it's already at or above the censoring point.
Not a clone of the actual hyz20/CWM repo (different label formulation, and CWM needs
torch 1.6.0) — a from-scratch encoding of its core idea (censored watch time, not a
binary threshold) using this repo's own proven causal features + LightGBM. Reuses
experiments/data_causal.py's feature engineering (same rows, same causal ordering)
but re-reads the raw CSVs directly to pull play_time_ms too, mirroring the
re-read-rather-than-edit convention used by experiments/data_mt.py. Train/valid
only; test never touched.
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

from starter_kit.data import SPLITS, _bucket_edges
from starter_kit.evaluate import evaluate
from experiments.data_causal import build_causal_features, NUMERIC_FIELDS, CAT_FIELDS

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LINEAR_BLEND_BEST = 0.6359
LGB_BEST = 0.6306
CAP = 1.0  # watch_ratio censoring point: "at least fully watched"

# best tuned LightGBM hyperparams (experiments/tune_lgb.py, [[project-best-model-lgb]]),
# reused verbatim so the only thing that changes vs. the proven ranker is loss/target.
LGB_PARAMS = {'learning_rate': 0.12, 'num_leaves': 255, 'min_data_in_leaf': 100,
              'feature_fraction': 0.5, 'bagging_fraction': 0.8, 'bagging_freq': 1,
              'lambda_l1': 0.0, 'lambda_l2': 10.0, 'max_depth': 5, 'min_gain_to_split': 0.01,
              'verbosity': -1, 'seed': 0}


def load_rows_with_watch(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows, play_ms = [], []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r['long_view'] != '0' else 0,
                             int(r['time_ms']), int(r['hourmin'])))
                play_ms.append(float(r['play_time_ms']))
    return rows, np.array(play_ms, dtype=np.float64)


def encode_with_watch(data_dir, smooth=5.0):
    rows, play_ms = load_rows_with_watch(data_dir)
    feat = build_causal_features(rows, smooth=smooth)

    dates = np.array([x[0] for x in rows], dtype=np.int64)
    durations = np.array([x[5] for x in rows], dtype=np.float32)
    labels = np.array([x[6] for x in rows], dtype=np.float32)
    users_all = np.array([x[1] for x in rows], dtype=object)
    authors_all = np.array([x[3] for x in rows], dtype=object)
    tabs_all = np.array([x[4] for x in rows], dtype=object)
    watch_ratio = play_ms / np.maximum(durations, 1.0)

    lo_tr, hi_tr = SPLITS['train']
    train_mask = (dates >= lo_tr) & (dates <= hi_tr)
    edges = _bucket_edges(durations[train_mask].tolist())
    dur_bucket_all = np.searchsorted(edges, durations).astype(np.int32)

    author_vocab = {v: i for i, v in enumerate(pd.unique(authors_all[train_mask]))}
    tab_vocab = {v: i for i, v in enumerate(pd.unique(tabs_all[train_mask]))}
    author_unk, tab_unk = len(author_vocab), len(tab_vocab)
    author_idx_all = pd.Series(authors_all).map(author_vocab).fillna(author_unk).to_numpy(dtype=np.int32)
    tab_idx_all = pd.Series(tabs_all).map(tab_vocab).fillna(tab_unk).to_numpy(dtype=np.int32)

    causal_cols = [f for f in NUMERIC_FIELDS if f not in ('duration_ms', 'hour')]
    feature_names = NUMERIC_FIELDS + ['dow'] + CAT_FIELDS

    enc = {}
    for name, (lo, hi) in SPLITS.items():
        idx = np.nonzero((dates >= lo) & (dates <= hi))[0]
        num = np.column_stack([feat[f][idx] for f in causal_cols] +
                               [durations[idx], feat['hour'][idx]]).astype(np.float32)
        cat = np.column_stack([author_idx_all[idx], tab_idx_all[idx], dur_bucket_all[idx]]).astype(np.int32)
        enc[name] = {'num': num, 'dow': feat['dow'][idx], 'cat': cat,
                      'y': labels[idx], 'watch_ratio': watch_ratio[idx],
                      'users': users_all[idx].tolist()}
    return enc, feature_names


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['watch_ratio'], d['users']


def make_tobit_objective(censored):
    def fobj(preds, dtrain):
        y = dtrain.get_label()
        target = np.where(censored, CAP, y)
        under = preds < target
        grad = np.where(censored, np.where(under, preds - CAP, 0.0), preds - y)
        hess = np.where(censored, np.where(under, 1.0, 1e-6), 1.0)
        return grad, hess
    return fobj


def make_tobit_feval(censored):
    def feval(preds, dtrain):
        y = dtrain.get_label()
        target = np.where(censored, CAP, y)
        under = preds < target
        err = np.where(censored, np.where(under, preds - CAP, 0.0), preds - y)
        return 'tobit_loss', float(np.mean(err ** 2)), False
    return feval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("=== building causal features + watch_ratio target ===")
    t0 = time.time()
    enc, feature_names = encode_with_watch(a.data_dir)
    print(f"  built in {time.time()-t0:.1f}s")

    Xtr, ytr, rtr, utr = build_matrix(enc, 'train')
    Xva, yva, rva, uva = build_matrix(enc, 'valid')
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]
    censored_tr = rtr >= CAP
    print(f"  censored fraction: train {censored_tr.mean():.3f}")

    group_map = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr], dtype=np.float32)

    dtrain = lgb.Dataset(Xtr, label=np.clip(rtr, 0, CAP * 3), weight=w_tr,
                          feature_name=feature_names, categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva, label=np.clip(rva, 0, CAP * 3), feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)

    print("\n=== iteration W: censored watch-time regression (Tobit-style LightGBM) ===")
    t0 = time.time()
    censored_va = rva >= CAP
    bst = lgb.train(dict(LGB_PARAMS, objective=make_tobit_objective(censored_tr)), dtrain,
                     num_boost_round=500, valid_sets=[dvalid],
                     feval=make_tobit_feval(censored_va),
                     callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    print(f"  trained {bst.best_iteration} rounds in {time.time()-t0:.1f}s")

    scores_va = bst.predict(Xva, num_iteration=bst.best_iteration)
    metrics = evaluate(uva, yva, scores_va)
    print(f"  -> valid GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
          f"primary {metrics['primary']:.4f}")
    print(f"delta vs baseline: {metrics['primary']-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs tuned LightGBM ranker ({LGB_BEST}): {metrics['primary']-LGB_BEST:+.4f}")
    print(f"delta vs linear blend best ({LINEAR_BLEND_BEST}): {metrics['primary']-LINEAR_BLEND_BEST:+.4f}")

    importance = dict(sorted(zip(feature_names, bst.feature_importance('gain').tolist()),
                              key=lambda kv: -kv[1]))

    np.save(os.path.join(a.out_dir, 'cwm_valid_scores.npy'), scores_va)
    np.save(os.path.join(a.out_dir, 'cwm_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'cwm_valid_labels.npy'), yva)
    with open('logs/iteration_W_cwm.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'lgb_ranker_best': LGB_BEST,
                    'linear_blend_best': LINEAR_BLEND_BEST, 'censored_frac_train': float(censored_tr.mean()),
                    'cwm_valid': metrics, 'best_iteration': bst.best_iteration,
                    'feature_importance_gain': importance}, fh, indent=2, default=float)
    print("logged to logs/iteration_W_cwm.json")


if __name__ == '__main__':
    main()
