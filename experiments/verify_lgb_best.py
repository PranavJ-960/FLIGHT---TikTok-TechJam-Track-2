"""Robustness check for iteration J's winning config: retrain across several
lightgbm seeds to confirm the gain isn't a lucky fit to one seed. Train/valid only.
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
BEST_PARAMS = {'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 20,
               'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'lambda_l1': 0.1,
               'lambda_l2': 1.0, 'max_depth': 5, 'min_gain_to_split': 0.01}


def build_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    os.makedirs(a.out_dir, exist_ok=True)

    enc, feature_names = encode_causal(a.data_dir)
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

    group_map = dict(zip(*np.unique(utr_s, return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr_s], dtype=np.float32)

    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, weight=w_tr, group=group_tr, feature_name=feature_names,
                          categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)

    print(f"=== verifying best config across seeds {seeds} ===")
    results = []
    all_scores = []
    for seed in seeds:
        run_params = dict(BEST_PARAMS, objective='lambdarank', metric='ndcg', eval_at=[5],
                           verbosity=-1, seed=seed)
        t0 = time.time()
        bst = lgb.train(run_params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        scores_va = bst.predict(Xva, num_iteration=bst.best_iteration)
        m = evaluate(uva, yva, scores_va)
        print(f"  seed={seed} primary={m['primary']:.4f} (delta vs baseline "
              f"{m['primary']-BASELINE_VALID['primary']:+.4f}) rounds={bst.best_iteration} "
              f"({time.time()-t0:.1f}s)")
        results.append({'seed': seed, 'valid': m, 'best_iteration': bst.best_iteration})
        all_scores.append(scores_va)

    primaries = [r['valid']['primary'] for r in results]
    print(f"\nmean primary={np.mean(primaries):.4f}  std={np.std(primaries):.4f}  "
          f"min={min(primaries):.4f}  max={max(primaries):.4f}")
    print(f"mean delta vs baseline: {np.mean(primaries)-BASELINE_VALID['primary']:+.4f}")

    # iteration K: multi-seed bagging (same architecture, different seeds — cheap since we
    # already trained these; ranks/scores averaged, not just metrics)
    norm_scores = []
    for s in all_scores:
        norm_scores.append((s - s.min()) / (s.max() - s.min() + 1e-12))
    bagged = np.mean(norm_scores, axis=0)
    bag_metrics = evaluate(uva, yva, bagged)
    print(f"\n=== iteration K: bagged ensemble ({len(seeds)} seeds, same config) ===")
    print(f"  bagged primary={bag_metrics['primary']:.4f} (delta vs baseline "
          f"{bag_metrics['primary']-BASELINE_VALID['primary']:+.4f}, delta vs mean single-seed "
          f"{bag_metrics['primary']-np.mean(primaries):+.4f})")

    np.save(os.path.join(a.out_dir, 'lgb_bagged_valid_scores.npy'), bagged)
    np.save(os.path.join(a.out_dir, 'lgb_bagged_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'lgb_bagged_valid_labels.npy'), yva)

    os.makedirs('logs', exist_ok=True)
    with open('logs/iteration_J_lgb_robustness.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'best_params': BEST_PARAMS,
                    'results': results, 'mean_primary': float(np.mean(primaries)),
                    'std_primary': float(np.std(primaries)),
                    'bagged_valid': bag_metrics}, fh, indent=2, default=float)
    print("logged to logs/iteration_J_lgb_robustness.json")


if __name__ == '__main__':
    main()
