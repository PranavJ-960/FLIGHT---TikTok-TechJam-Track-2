"""Iteration E/F/G: hyperparameter-tune D (k, lr — never swept, every prior run reused
the plain FM baseline's defaults), then ensemble D across seeds at the winning config, and
compare the stacked result (tuned + ensembled) against D as the new best-so-far. Train/valid
only — test is never unpacked or evaluated, per the hard rule.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from starter_kit.data import FIELDS
from starter_kit.evaluate import evaluate
from experiments.data_seq import encode_seq
from experiments.fm_lib import SharedEmbeddingFM, pointwise_seq_contribution, predict_seq, sigmoid

MAIN = 'long_view'
VIDEO_FIELD = FIELDS.index('video_id')

# D, default hyperparams (k=16, lr=0.001), valid primary per seed — our best-so-far baseline.
D_DEFAULT_VALID = {0: 0.6027, 1: 0.6011, 2: 0.6019}
BASELINE_VALID_BY_SEED = {
    0: {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015},
    1: {'GAUC': 0.6674, 'nDCG@5': 0.5361, 'primary': 0.6018},
    2: {'GAUC': 0.6671, 'nDCG@5': 0.5351, 'primary': 0.6011},
}


def train_D(enc, dim, k, lr, seed, bs=8192, epochs=40, patience=4, verbose=True):
    """Train one D model on pre-loaded `enc`. Returns (valid_metrics, valid_probs)."""
    Xtr, Htr, Mtr, ytr, utr = enc['train']
    Xva, Hva, Mva, yva, uva = enc['valid']

    model = SharedEmbeddingFM(dim + 1, tasks=[MAIN], k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(utr)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            b_idx = idx[i:i + bs]
            model.begin_batch()
            losses.append(pointwise_seq_contribution(
                model, MAIN, Xtr[b_idx], Htr[b_idx], Mtr[b_idx], ytr[b_idx], len(b_idx), VIDEO_FIELD))
            model.commit()
        va = evaluate(uva, yva, predict_seq(model, MAIN, Xva, Hva, Mva, VIDEO_FIELD))
        if verbose:
            print(f"    epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (model.V.copy(), {t: w.copy() for t, w in model.W.items()},
                           {t: bb for t, bb in model.b.items()})
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep}")
                break
    V, W, b = best_state
    model.V, model.W, model.b = V, {t: w.copy() for t, w in W.items()}, {t: bb for t, bb in b.items()}
    z = predict_seq(model, MAIN, Xva, Hva, Mva, VIDEO_FIELD)
    return evaluate(uva, yva, sigmoid(z)), sigmoid(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--tune_seed', type=int, default=0)
    ap.add_argument('--ensemble_seeds', default='0,1,2')
    a = ap.parse_args()
    ensemble_seeds = [int(s) for s in a.ensemble_seeds.split(',')]

    print(f"loading sequence-encoded data (max_len={a.max_len}) ...")
    t0 = time.time()
    enc, dim, pad_idx = encode_seq(a.data_dir, max_len=a.max_len)
    print(f"  loaded in {time.time()-t0:.1f}s")
    Xva, Hva, Mva, yva, uva = enc['valid']

    log = {'lr_sweep': {}, 'k_sweep': {}, 'ensemble_members': {}}

    # ---- iteration E: lr sweep at k=16, tune_seed only ----
    print(f"\n=== iteration E: lr sweep (k=16, seed={a.tune_seed}) ===")
    lr_grid = [0.0006, 0.001, 0.0015, 0.002]
    lr_results = {}
    for lr in lr_grid:
        print(f"  -- lr={lr} --")
        va, _ = train_D(enc, dim, k=16, lr=lr, seed=a.tune_seed, verbose=False)
        lr_results[lr] = va['primary']
        log['lr_sweep'][lr] = va
        print(f"     valid primary {va['primary']:.4f}")
    best_lr = max(lr_results, key=lr_results.get)
    print(f"  best lr = {best_lr} (valid primary {lr_results[best_lr]:.4f})")

    # ---- iteration E cont'd: k sweep at best_lr, tune_seed only ----
    print(f"\n=== iteration E: k sweep (lr={best_lr}, seed={a.tune_seed}) ===")
    k_grid = [8, 16, 24, 32]
    k_results = {16: lr_results[best_lr]} if best_lr == 0.001 else {}
    for k in k_grid:
        if k in k_results:
            continue
        print(f"  -- k={k} --")
        va, _ = train_D(enc, dim, k=k, lr=best_lr, seed=a.tune_seed, verbose=False)
        k_results[k] = va['primary']
        log['k_sweep'][k] = va
        print(f"     valid primary {va['primary']:.4f}")
    best_k = max(k_results, key=k_results.get)
    print(f"  best k = {best_k} (valid primary {k_results[best_k]:.4f})")

    best_config = {'k': best_k, 'lr': best_lr}
    base_delta = k_results[best_k] - BASELINE_VALID_BY_SEED[a.tune_seed]['primary']
    d_delta = k_results[best_k] - D_DEFAULT_VALID[a.tune_seed]
    print(f"\nbest single config: k={best_k} lr={best_lr} -> valid primary {k_results[best_k]:.4f} "
          f"(delta vs baseline {base_delta:+.4f}, vs D-default {d_delta:+.4f})")

    # ---- iteration F/G: ensemble at best_config across ensemble_seeds ----
    print(f"\n=== iteration F/G: ensemble at k={best_k} lr={best_lr} across seeds {ensemble_seeds} ===")
    probs = []
    for seed in ensemble_seeds:
        print(f"  -- training member seed={seed} --")
        va, p = train_D(enc, dim, k=best_k, lr=best_lr, seed=seed, verbose=False)
        probs.append(p)
        log['ensemble_members'][seed] = va
        print(f"     member valid primary {va['primary']:.4f}")

    avg_probs = np.mean(probs, axis=0)
    ens_metrics = evaluate(uva, yva, avg_probs)
    print(f"\nensembled ({len(ensemble_seeds)} seeds) valid: GAUC {ens_metrics['GAUC']:.4f} "
          f"nDCG@5 {ens_metrics['nDCG@5']:.4f} primary {ens_metrics['primary']:.4f}")

    baseline_mean = np.mean([BASELINE_VALID_BY_SEED[s]['primary'] for s in ensemble_seeds])
    d_default_mean = np.mean([D_DEFAULT_VALID[s] for s in ensemble_seeds])
    print(f"\n{'':22s} {'valid primary':>14s} {'delta vs baseline':>19s} {'delta vs D-default':>20s}")
    print(f"{'baseline (mean)':22s} {baseline_mean:14.4f} {'—':>19s} {'—':>20s}")
    print(f"{'D-default (mean)':22s} {d_default_mean:14.4f} {d_default_mean-baseline_mean:+19.4f} {'—':>20s}")
    print(f"{'tuned single (seed'+str(a.tune_seed)+')':22s} {k_results[best_k]:14.4f} "
          f"{base_delta:+19.4f} {d_delta:+20.4f}")
    print(f"{'tuned+ensembled':22s} {ens_metrics['primary']:14.4f} "
          f"{ens_metrics['primary']-baseline_mean:+19.4f} {ens_metrics['primary']-d_default_mean:+20.4f}")

    os.makedirs('logs', exist_ok=True)
    out_path = 'logs/iteration_EFG_tune_ensemble_D.json'
    with open(out_path, 'w') as fh:
        json.dump({
            'best_config': best_config,
            'lr_grid_results': {str(k): v for k, v in lr_results.items()},
            'k_grid_results': {str(k): v for k, v in k_results.items()},
            'ensemble_seeds': ensemble_seeds,
            'ensemble_metrics': ens_metrics,
            'baseline_valid_by_seed': BASELINE_VALID_BY_SEED,
            'd_default_valid_by_seed': D_DEFAULT_VALID,
            'detail': log,
        }, fh, indent=2, default=float)
    print(f"\nlogged to {out_path}")


if __name__ == '__main__':
    main()
