"""Recompute the tuned FM ensemble (k=16, lr=0.0006, seeds 0-2) and save its valid
probabilities for blending. Standalone process — kept separate from lightgbm (see
experiments/train_lgb.py's docstring: torch/MPS + lightgbm in one process deadlocks).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from starter_kit.evaluate import evaluate
from experiments.data_seq import encode_seq
from experiments.train_torch import to_device_tensors, train_D_torch
from experiments.fm_torch import best_device

BEST_CONFIG = {'k': 16, 'lr': 0.0006}
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    device = best_device()
    print(f"device = {device}")
    enc, dim, pad_idx = encode_seq(a.data_dir, max_len=20)
    tensors_tr = to_device_tensors(enc, 'train', device)
    tensors_va = to_device_tensors(enc, 'valid', device)

    probs = []
    for seed in SEEDS:
        t0 = time.time()
        va, p = train_D_torch(tensors_tr, tensors_va, dim, k=BEST_CONFIG['k'], lr=BEST_CONFIG['lr'],
                               seed=seed, device=device, verbose=False)
        print(f"  seed={seed} valid primary {va['primary']:.4f} ({time.time()-t0:.1f}s)")
        probs.append(p)

    avg = np.mean(probs, axis=0)
    _, _, _, yva, uva = enc['valid']
    metrics = evaluate(uva, yva, avg)
    print(f"-> ensemble valid GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
          f"primary {metrics['primary']:.4f}")

    np.save(os.path.join(a.out_dir, 'fm_valid_scores.npy'), avg)
    np.save(os.path.join(a.out_dir, 'fm_valid_users.npy'), np.array(uva))
    np.save(os.path.join(a.out_dir, 'fm_valid_labels.npy'), yva)
    with open('logs/iteration_FM_ensemble_for_blend.json', 'w') as fh:
        json.dump({'config': BEST_CONFIG, 'seeds': SEEDS, 'ensemble_valid': metrics}, fh, indent=2, default=float)
    print(f"saved scores to {a.out_dir}/")


if __name__ == '__main__':
    main()
