"""Iteration E/F/G, GPU + CPU-parallel version: hyperparameter-sweep D on MPS
(each config trained in its own process, so the sweep uses both the GPU and
several CPU cores at once), then ensemble across seeds at the winning config,
and compare against D as the best-so-far. Train/valid only — every worker loads
data itself and never touches the test split.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BASELINE_VALID_BY_SEED = {
    0: {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015},
    1: {'GAUC': 0.6674, 'nDCG@5': 0.5361, 'primary': 0.6018},
    2: {'GAUC': 0.6671, 'nDCG@5': 0.5351, 'primary': 0.6011},
}
D_NUMPY_VALID = {0: 0.6027, 1: 0.6011, 2: 0.6019}   # iteration D, hand-rolled NumPy/CPU version


def _worker(job):
    """Runs in its own process: load data + train one (k, lr, seed) config on MPS."""
    data_dir, max_len, k, lr, seed, epochs, patience, want_probs = job
    from experiments.data_seq import encode_seq
    from experiments.train_torch import to_device_tensors, train_D_torch
    from experiments.fm_torch import best_device
    device = best_device()
    enc, dim, pad_idx = encode_seq(data_dir, max_len=max_len)
    tensors_tr = to_device_tensors(enc, 'train', device)
    tensors_va = to_device_tensors(enc, 'valid', device)
    va, probs = train_D_torch(tensors_tr, tensors_va, dim, k=k, lr=lr, epochs=epochs,
                               patience=patience, seed=seed, device=device, verbose=False)
    return {'k': k, 'lr': lr, 'seed': seed, 'valid': va,
            'probs': probs if want_probs else None}


def run_stage(jobs, max_workers, label):
    print(f"\n--- {label}: {len(jobs)} configs, up to {max_workers} in parallel ---")
    t0 = time.time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_worker, jobs):
            print(f"    k={res['k']:3d} lr={res['lr']:.4f} seed={res['seed']} "
                  f"-> valid primary {res['valid']['primary']:.4f}")
            results.append(res)
    print(f"  stage wall time: {time.time()-t0:.1f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--tune_seed', type=int, default=0)
    ap.add_argument('--ensemble_seeds', default='0,1,2')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--max_workers', type=int, default=4)
    a = ap.parse_args()
    ensemble_seeds = [int(s) for s in a.ensemble_seeds.split(',')]

    t_start = time.time()

    # ---- E: lr sweep at k=16 ----
    lr_grid = [0.0006, 0.001, 0.0015, 0.002, 0.003]
    jobs = [(a.data_dir, a.max_len, 16, lr, a.tune_seed, a.epochs, a.patience, False) for lr in lr_grid]
    lr_res = run_stage(jobs, a.max_workers, "lr sweep (k=16)")
    best_lr = max(lr_res, key=lambda r: r['valid']['primary'])['lr']
    print(f"best lr = {best_lr}")

    # ---- E: k sweep at best_lr ----
    k_grid = [8, 16, 24, 32, 48]
    jobs = [(a.data_dir, a.max_len, k, best_lr, a.tune_seed, a.epochs, a.patience, False) for k in k_grid]
    k_res = run_stage(jobs, a.max_workers, f"k sweep (lr={best_lr})")
    best_k = max(k_res, key=lambda r: r['valid']['primary'])['k']
    best_single = max(r['valid']['primary'] for r in k_res)
    print(f"best k = {best_k} -> best single-config valid primary {best_single:.4f}")

    # ---- F/G: ensemble at (best_k, best_lr) across seeds ----
    jobs = [(a.data_dir, a.max_len, best_k, best_lr, s, a.epochs, a.patience, True) for s in ensemble_seeds]
    ens_res = run_stage(jobs, a.max_workers, f"ensemble members (k={best_k} lr={best_lr})")
    probs = [r['probs'] for r in ens_res]
    avg_probs = np.mean(probs, axis=0)

    from experiments.data_seq import encode_seq
    enc, _, _ = encode_seq(a.data_dir, max_len=a.max_len)
    _, _, _, yva, uva = enc['valid']
    from starter_kit.evaluate import evaluate
    ens_metrics = evaluate(uva, yva, avg_probs)

    baseline_mean = np.mean([BASELINE_VALID_BY_SEED[s]['primary'] for s in ensemble_seeds])
    d_numpy_mean = np.mean([D_NUMPY_VALID[s] for s in ensemble_seeds])
    print(f"\n{'':26s} {'valid primary':>14s} {'delta vs baseline':>19s} {'delta vs D (numpy)':>20s}")
    print(f"{'baseline (mean)':26s} {baseline_mean:14.4f} {'—':>19s} {'—':>20s}")
    print(f"{'D, numpy (mean)':26s} {d_numpy_mean:14.4f} {d_numpy_mean-baseline_mean:+19.4f} {'—':>20s}")
    print(f"{'tuned single (seed'+str(a.tune_seed)+')':26s} {best_single:14.4f} "
          f"{best_single-BASELINE_VALID_BY_SEED[a.tune_seed]['primary']:+19.4f} "
          f"{best_single-D_NUMPY_VALID[a.tune_seed]:+20.4f}")
    print(f"{'tuned + ensembled':26s} {ens_metrics['primary']:14.4f} "
          f"{ens_metrics['primary']-baseline_mean:+19.4f} {ens_metrics['primary']-d_numpy_mean:+20.4f}")

    os.makedirs('logs', exist_ok=True)
    out_path = 'logs/iteration_EFG_torch_tune_ensemble.json'
    with open(out_path, 'w') as fh:
        json.dump({
            'best_config': {'k': best_k, 'lr': best_lr},
            'lr_sweep': [{'lr': r['lr'], 'valid': r['valid']} for r in lr_res],
            'k_sweep': [{'k': r['k'], 'valid': r['valid']} for r in k_res],
            'ensemble_seeds': ensemble_seeds,
            'ensemble_members': [{'seed': r['seed'], 'valid': r['valid']} for r in ens_res],
            'ensemble_metrics': ens_metrics,
            'baseline_valid_by_seed': BASELINE_VALID_BY_SEED,
            'd_numpy_valid_by_seed': D_NUMPY_VALID,
        }, fh, indent=2, default=float)
    print(f"\nlogged to {out_path}")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
