"""Iteration O: blend LightGBM (bagged), XGBoost, and CatBoost — three genuinely
different GBT implementations on the same causal features, unlike the FM+LightGBM
blend (which failed because FM's errors were a subset of LightGBM's, not
complementary). Grid-search 2-way and 3-way weights on valid. Numpy-only, run after
train_lgb.py/verify_lgb_best.py, train_xgb.py, and train_catboost.py have all saved
their outputs/*.npy files.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from starter_kit.evaluate import evaluate

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}


def load(out_dir, prefix):
    scores = np.load(os.path.join(out_dir, f'{prefix}_valid_scores.npy'))
    users = np.load(os.path.join(out_dir, f'{prefix}_valid_users.npy'), allow_pickle=True)
    labels = np.load(os.path.join(out_dir, f'{prefix}_valid_labels.npy'))
    return scores, list(users), labels


def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()

    lgb_scores, lgb_users, lgb_labels = load(a.out_dir, 'lgb_bagged')
    xgb_scores, xgb_users, xgb_labels = load(a.out_dir, 'xgb')
    cat_scores, cat_users, cat_labels = load(a.out_dir, 'catboost')

    assert lgb_users == xgb_users == cat_users, "row order mismatch across models"
    assert np.array_equal(lgb_labels, xgb_labels) and np.array_equal(lgb_labels, cat_labels)
    uva, yva = lgb_users, lgb_labels

    m_lgb = evaluate(uva, yva, lgb_scores)
    m_xgb = evaluate(uva, yva, xgb_scores)
    m_cat = evaluate(uva, yva, cat_scores)
    print(f"LightGBM (bagged) : primary={m_lgb['primary']:.4f}")
    print(f"XGBoost           : primary={m_xgb['primary']:.4f}")
    print(f"CatBoost          : primary={m_cat['primary']:.4f}")

    lgb_n, xgb_n, cat_n = norm(lgb_scores), norm(xgb_scores), norm(cat_scores)

    print("\n=== 2-way blends ===")
    best = {'primary': -1}
    results_2way = {}
    for name_a, s_a, name_b, s_b in [('lgb', lgb_n, 'xgb', xgb_n),
                                       ('lgb', lgb_n, 'cat', cat_n),
                                       ('xgb', xgb_n, 'cat', cat_n)]:
        for w in np.arange(0.0, 1.01, 0.1):
            blended = w * s_a + (1 - w) * s_b
            m = evaluate(uva, yva, blended)
            key = f"{name_a}={w:.1f}/{name_b}={1-w:.1f}"
            results_2way[key] = m['primary']
            if m['primary'] > best['primary']:
                best = {'primary': m['primary'], 'combo': key, 'metrics': m}
    for k, v in sorted(results_2way.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:22s} primary={v:.4f}")

    print("\n=== 3-way blends (weights on a simplex grid) ===")
    best3 = {'primary': -1}
    results_3way = {}
    steps = [i / 10 for i in range(11)]
    for w_lgb in steps:
        for w_xgb in steps:
            w_cat = round(1.0 - w_lgb - w_xgb, 2)
            if w_cat < -1e-9 or w_cat > 1 + 1e-9:
                continue
            w_cat = max(0.0, w_cat)
            blended = w_lgb * lgb_n + w_xgb * xgb_n + w_cat * cat_n
            m = evaluate(uva, yva, blended)
            key = f"lgb={w_lgb:.1f}/xgb={w_xgb:.1f}/cat={w_cat:.1f}"
            results_3way[key] = m['primary']
            if m['primary'] > best3['primary']:
                best3 = {'primary': m['primary'], 'combo': key, 'metrics': m}
    for k, v in sorted(results_3way.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {k:26s} primary={v:.4f}")

    print(f"\n{'model/blend':30s} {'valid primary':>14s} {'delta vs baseline':>19s} {'delta vs LGB alone':>20s}")
    print(f"{'baseline':30s} {BASELINE_VALID['primary']:14.4f} {'—':>19s} {'—':>20s}")
    print(f"{'LightGBM (bagged)':30s} {m_lgb['primary']:14.4f} "
          f"{m_lgb['primary']-BASELINE_VALID['primary']:+19.4f} {'—':>20s}")
    print(f"{'XGBoost':30s} {m_xgb['primary']:14.4f} "
          f"{m_xgb['primary']-BASELINE_VALID['primary']:+19.4f} {m_xgb['primary']-m_lgb['primary']:+20.4f}")
    print(f"{'CatBoost':30s} {m_cat['primary']:14.4f} "
          f"{m_cat['primary']-BASELINE_VALID['primary']:+19.4f} {m_cat['primary']-m_lgb['primary']:+20.4f}")
    print(f"{'best 2-way (' + best['combo'] + ')':30s} {best['metrics']['primary']:14.4f} "
          f"{best['metrics']['primary']-BASELINE_VALID['primary']:+19.4f} "
          f"{best['metrics']['primary']-m_lgb['primary']:+20.4f}")
    print(f"{'best 3-way (' + best3['combo'] + ')':30s} {best3['metrics']['primary']:14.4f} "
          f"{best3['metrics']['primary']-BASELINE_VALID['primary']:+19.4f} "
          f"{best3['metrics']['primary']-m_lgb['primary']:+20.4f}")

    os.makedirs('logs', exist_ok=True)
    with open('logs/iteration_O_blend_multi.json', 'w') as fh:
        json.dump({
            'baseline_valid': BASELINE_VALID,
            'lgb_valid': m_lgb, 'xgb_valid': m_xgb, 'catboost_valid': m_cat,
            'best_2way': best, 'best_3way': best3,
        }, fh, indent=2, default=float)
    print("\nlogged to logs/iteration_O_blend_multi.json")


if __name__ == '__main__':
    main()
