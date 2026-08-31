"""Iteration L: LightGBM on causal features + FM's learned user/video embeddings as
extra input features (representation-level stacking — different from blend_scores.py's
prediction-level blending, which didn't help). Requires experiments/save_fm_embeddings.py
to have already run and saved outputs/fm_{user,video}_emb_{train,valid}.npy. Train/valid
only; standalone process (no torch import — see experiments/train_lgb.py's docstring on
the torch/lightgbm deadlock).
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
LGB_BEST_SO_FAR = 0.6264   # 5-seed bagged, causal features only (iteration K)
BEST_PARAMS = {'learning_rate': 0.2, 'num_leaves': 63, 'min_data_in_leaf': 50,
               'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'lambda_l1': 1.0,
               'lambda_l2': 0.0, 'max_depth': 5, 'min_gain_to_split': 0.01}


def build_matrix(enc, split, out_dir):
    d = enc[split]
    emb_users = np.load(os.path.join(out_dir, f'fm_emb_{split}_users.npy'), allow_pickle=True)
    assert list(emb_users) == d['users'], f"row order mismatch between FM embeddings and causal features ({split})"
    user_emb = np.load(os.path.join(out_dir, f'fm_user_emb_{split}.npy'))
    video_emb = np.load(os.path.join(out_dir, f'fm_video_emb_{split}.npy'))
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32),
                         user_emb, video_emb], axis=1)
    return X, d['y'], d['users']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    print("=== building causal features ===")
    enc, feature_names = encode_causal(a.data_dir)
    emb_names = [f'fm_user_emb_{i}' for i in range(16)] + [f'fm_video_emb_{i}' for i in range(16)]
    feature_names = feature_names + emb_names

    Xtr, ytr, utr = build_matrix(enc, 'train', a.out_dir)
    Xva, yva, uva = build_matrix(enc, 'valid', a.out_dir)
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]

    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    group_tr = np.unique([utr[i] for i in order_tr], return_counts=True)[1]
    group_va = np.unique([uva[i] for i in order_va], return_counts=True)[1]

    dtrain = lgb.Dataset(Xtr_s, label=ytr_s, group=group_tr, feature_name=feature_names,
                          categorical_feature=cat_idx)
    dvalid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, feature_name=feature_names,
                          categorical_feature=cat_idx, reference=dtrain)

    print(f"\n=== iteration L: LightGBM + FM embeddings, seeds {seeds} ===")
    results, all_scores = [], []
    for seed in seeds:
        run_params = dict(BEST_PARAMS, objective='lambdarank', metric='ndcg', eval_at=[5],
                           verbosity=-1, seed=seed)
        t0 = time.time()
        bst = lgb.train(run_params, dtrain, num_boost_round=500, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        scores_va = bst.predict(Xva, num_iteration=bst.best_iteration)
        m = evaluate(uva, yva, scores_va)
        print(f"  seed={seed} primary={m['primary']:.4f} rounds={bst.best_iteration} ({time.time()-t0:.1f}s)")
        results.append({'seed': seed, 'valid': m, 'best_iteration': bst.best_iteration})
        all_scores.append(scores_va)
        if seed == seeds[0]:
            importance = dict(sorted(zip(feature_names, bst.feature_importance('gain').tolist()),
                                      key=lambda kv: -kv[1])[:15])

    primaries = [r['valid']['primary'] for r in results]
    print(f"\nmean primary={np.mean(primaries):.4f}  std={np.std(primaries):.4f}")
    print(f"delta vs baseline: {np.mean(primaries)-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs causal-features-only bagged best ({LGB_BEST_SO_FAR}): "
          f"{np.mean(primaries)-LGB_BEST_SO_FAR:+.4f}")
    print("top-15 feature importances (gain, seed 0):", importance)

    norm = [(s - s.min()) / (s.max() - s.min() + 1e-12) for s in all_scores]
    bagged = np.mean(norm, axis=0)
    bag_metrics = evaluate(uva, yva, bagged)
    print(f"\nbagged ({len(seeds)} seeds) primary={bag_metrics['primary']:.4f} "
          f"(delta vs baseline {bag_metrics['primary']-BASELINE_VALID['primary']:+.4f}, "
          f"delta vs causal-only-bagged {bag_metrics['primary']-LGB_BEST_SO_FAR:+.4f})")

    os.makedirs('logs', exist_ok=True)
    with open('logs/iteration_L_lgb_stacked.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'lgb_causal_only_bagged': LGB_BEST_SO_FAR,
                    'params': BEST_PARAMS, 'results': results,
                    'mean_primary': float(np.mean(primaries)), 'std_primary': float(np.std(primaries)),
                    'bagged_valid': bag_metrics, 'top_feature_importance': importance},
                   fh, indent=2, default=float)
    print("logged to logs/iteration_L_lgb_stacked.json")


if __name__ == '__main__':
    main()
