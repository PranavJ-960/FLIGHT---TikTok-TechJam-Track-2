"""Iteration I: blend the saved LightGBM and FM-ensemble valid predictions (alpha
sweep). Numpy-only — no torch or lightgbm import needed here, run after both
experiments/train_lgb.py and experiments/save_fm_ensemble.py have saved their
outputs/*.npy files.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from starter_kit.evaluate import evaluate

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()

    lgb_scores = np.load(os.path.join(a.out_dir, 'lgb_valid_scores.npy'))
    lgb_users = np.load(os.path.join(a.out_dir, 'lgb_valid_users.npy'), allow_pickle=True)
    lgb_labels = np.load(os.path.join(a.out_dir, 'lgb_valid_labels.npy'))
    fm_scores = np.load(os.path.join(a.out_dir, 'fm_valid_scores.npy'))
    fm_users = np.load(os.path.join(a.out_dir, 'fm_valid_users.npy'), allow_pickle=True)
    fm_labels = np.load(os.path.join(a.out_dir, 'fm_valid_labels.npy'))

    assert list(lgb_users) == list(fm_users), "row order mismatch between LightGBM and FM valid sets"
    assert np.array_equal(lgb_labels, fm_labels), "label mismatch between LightGBM and FM valid sets"
    uva, yva = lgb_users, lgb_labels

    lgb_metrics = evaluate(uva, yva, lgb_scores)
    fm_metrics = evaluate(uva, yva, fm_scores)

    fm_norm = (fm_scores - fm_scores.min()) / (fm_scores.max() - fm_scores.min() + 1e-12)
    lgb_norm = (lgb_scores - lgb_scores.min()) / (lgb_scores.max() - lgb_scores.min() + 1e-12)

    print("=== iteration I: blend FM ensemble + LightGBM (alpha sweep on valid) ===")
    results = {}
    best_alpha, best_metrics = None, None
    for alpha in np.arange(0.0, 1.01, 0.1):
        blended = alpha * fm_norm + (1 - alpha) * lgb_norm
        m = evaluate(uva, yva, blended)
        results[round(float(alpha), 2)] = m
        print(f"  alpha(FM)={alpha:.1f} -> valid primary {m['primary']:.4f}")
        if best_metrics is None or m['primary'] > best_metrics['primary']:
            best_alpha, best_metrics = round(float(alpha), 2), m

    print(f"\n{'model':26s} {'valid primary':>14s} {'delta vs baseline':>19s}")
    print(f"{'baseline (mean)':26s} {BASELINE_VALID['primary']:14.4f} {'—':>19s}")
    print(f"{'FM ensemble (tuned)':26s} {fm_metrics['primary']:14.4f} "
          f"{fm_metrics['primary']-BASELINE_VALID['primary']:+19.4f}")
    print(f"{'LightGBM (causal feats)':26s} {lgb_metrics['primary']:14.4f} "
          f"{lgb_metrics['primary']-BASELINE_VALID['primary']:+19.4f}")
    print(f"{'blend, alpha='+str(best_alpha):26s} {best_metrics['primary']:14.4f} "
          f"{best_metrics['primary']-BASELINE_VALID['primary']:+19.4f}")

    os.makedirs('logs', exist_ok=True)
    with open('logs/iteration_I_blend.json', 'w') as fh:
        json.dump({
            'baseline_valid': BASELINE_VALID,
            'fm_ensemble_valid': fm_metrics,
            'lgb_valid': lgb_metrics,
            'alpha_sweep': results,
            'best_blend': {'alpha_fm': best_alpha, 'metrics': best_metrics},
        }, fh, indent=2, default=float)
    print("\nlogged to logs/iteration_I_blend.json")


if __name__ == '__main__':
    main()
