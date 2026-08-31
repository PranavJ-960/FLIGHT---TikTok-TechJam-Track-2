"""Run iteration A (pairwise loss), B (multi-task auxiliary signals), or C (combined)
against the FM baseline, log results to logs/, and print a comparison table.

Usage:
  python3 experiments/run_iteration.py --iteration A
  python3 experiments/run_iteration.py --iteration all
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from starter_kit.data import load, encode, FIELDS
from starter_kit.evaluate import evaluate
from experiments.data_mt import load_mt, encode_mt, AUX_TASKS
from experiments.data_seq import encode_seq
from experiments.fm_lib import (
    SharedEmbeddingFM, pointwise_contribution, pairwise_contribution,
    build_user_pos_neg, sample_pairs, sigmoid,
    pointwise_seq_contribution, predict_seq,
)

MAIN = 'long_view'
HYPOTHESES = {
    'A': "Pairwise (BPR) loss on long_view directly optimizes within-user ranking, "
         "matching GAUC/nDCG@5, instead of pointwise logloss which optimizes calibration.",
    'B': "Auxiliary tasks (is_click/is_like/is_follow) sharing the embedding table give "
         "denser gradient signal than the sparse long_view label alone, regularizing V.",
    'C': "Effects of A and B are additive: pairwise ranking loss on long_view's head, "
         "auxiliary pointwise losses on shared embeddings, both feeding the same V.",
    'D': "A/B/C only change how the same user_id x video_id embedding is trained; "
         "attention over a user's actual watch history (DIN-style) gives the model "
         "genuinely new information the FM baseline has zero access to.",
}


def _restore_or_keep(best_state, model):
    if best_state is not None:
        V, W, b = best_state
        model.V = V
        model.W = {t: w.copy() for t, w in W.items()}
        model.b = {t: bb for t, bb in b.items()}


def _snapshot(model):
    return (model.V.copy(), {t: w.copy() for t, w in model.W.items()},
            {t: bb for t, bb in model.b.items()})


def run_A(data_dir, k=16, lr=0.001, bs=8192, epochs=40, patience=4, seed=0, verbose=True):
    splits = load(data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    # starter_kit's encode() returns all 3 splits together; enc['test'] is deliberately
    # never unpacked/evaluated here — dev only ever touches train/valid.

    model = SharedEmbeddingFM(dim, tasks=[MAIN], k=k, lr=lr, seed=seed)
    groups = build_user_pos_neg(ytr, utr)
    rng = np.random.default_rng(seed)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        losses = []
        for i in range(0, len(pos_idx), bs):
            Xp = Xtr[pos_idx[i:i + bs]]
            Xn = Xtr[neg_idx[i:i + bs]]
            model.begin_batch()
            losses.append(pairwise_contribution(model, MAIN, Xp, Xn, len(Xp)))
            model.commit()
        va = evaluate(uva, yva, model.predict(Xva, MAIN))
        history.append(va)
        if verbose:
            print(f"  epoch {ep:2d} | pair-loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_state = va['primary'], 0, _snapshot(model)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    _restore_or_keep(best_state, model)
    return {'valid': evaluate(uva, yva, model.predict(Xva, MAIN)),
            'epochs_run': len(history), 'history': history}


def run_B(data_dir, k=16, lr=0.001, bs=8192, epochs=40, patience=4, seed=0, verbose=True,
          aux_weight=None):
    splits = load_mt(data_dir)
    enc, dim = encode_mt(splits)
    Xtr, Ytr, utr = enc['train']
    Xva, Yva, uva = enc['valid']
    # enc['test'] deliberately never unpacked/evaluated here — dev only touches train/valid.
    tasks = [MAIN] + AUX_TASKS
    aux_weight = aux_weight if aux_weight is not None else 1.0 / len(AUX_TASKS)

    model = SharedEmbeddingFM(dim, tasks=tasks, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(utr)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            b_idx = idx[i:i + bs]
            X_b = Xtr[b_idx]
            model.begin_batch()
            main_loss = None
            for task in tasks:
                y_b = Ytr[task][b_idx]
                w = 1.0 if task == MAIN else aux_weight
                z, E, S = model.logits(X_b, task)
                g = ((sigmoid(z) - y_b) / len(y_b) * w).astype(np.float32)
                model.add_contribution(task, X_b, g, E, S)
                if task == MAIN:
                    main_loss = float(-np.mean(y_b * np.log(sigmoid(z) + 1e-9) +
                                                (1 - y_b) * np.log(1 - sigmoid(z) + 1e-9)))
            model.commit()
            losses.append(main_loss)
        va = evaluate(uva, Yva[MAIN], model.predict(Xva, MAIN))
        history.append(va)
        if verbose:
            print(f"  epoch {ep:2d} | main-loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_state = va['primary'], 0, _snapshot(model)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    _restore_or_keep(best_state, model)
    return {'valid': evaluate(uva, Yva[MAIN], model.predict(Xva, MAIN)),
            'epochs_run': len(history), 'history': history}


def run_C(data_dir, k=16, lr=0.001, bs=8192, epochs=40, patience=4, seed=0, verbose=True,
          aux_weight=None):
    splits = load_mt(data_dir)
    enc, dim = encode_mt(splits)
    Xtr, Ytr, utr = enc['train']
    Xva, Yva, uva = enc['valid']
    # enc['test'] deliberately never unpacked/evaluated here — dev only touches train/valid.
    tasks = [MAIN] + AUX_TASKS
    aux_weight = aux_weight if aux_weight is not None else 1.0 / len(AUX_TASKS)

    model = SharedEmbeddingFM(dim, tasks=tasks, k=k, lr=lr, seed=seed)
    groups = build_user_pos_neg(Ytr[MAIN], utr)
    rng = np.random.default_rng(seed)
    n = len(utr)
    steps_pw = math.ceil(n / bs)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(n)
        pos_idx, neg_idx = sample_pairs(groups, rng)
        steps_pair = math.ceil(len(pos_idx) / bs)
        steps = max(steps_pw, steps_pair)
        losses = []
        for s in range(steps):
            lo = (s % steps_pw) * bs
            b_idx = idx[lo:lo + bs]
            X_b = Xtr[b_idx]
            model.begin_batch()
            for task in AUX_TASKS:
                y_b = Ytr[task][b_idx]
                z, E, S = model.logits(X_b, task)
                g = ((sigmoid(z) - y_b) / len(y_b) * aux_weight).astype(np.float32)
                model.add_contribution(task, X_b, g, E, S)
            model.commit()

            lo2 = (s % steps_pair) * bs
            Xp = Xtr[pos_idx[lo2:lo2 + bs]]
            Xn = Xtr[neg_idx[lo2:lo2 + bs]]
            model.begin_batch()
            losses.append(pairwise_contribution(model, MAIN, Xp, Xn, len(Xp)))
            model.commit()
        va = evaluate(uva, Yva[MAIN], model.predict(Xva, MAIN))
        history.append(va)
        if verbose:
            print(f"  epoch {ep:2d} | pair-loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_state = va['primary'], 0, _snapshot(model)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    _restore_or_keep(best_state, model)
    return {'valid': evaluate(uva, Yva[MAIN], model.predict(Xva, MAIN)),
            'epochs_run': len(history), 'history': history}


def run_D(data_dir, k=16, lr=0.001, bs=8192, epochs=40, patience=4, seed=0, verbose=True,
          max_len=20):
    enc, dim, pad_idx = encode_seq(data_dir, max_len=max_len)
    Xtr, Htr, Mtr, ytr, utr = enc['train']
    Xva, Hva, Mva, yva, uva = enc['valid']
    # enc['test'] deliberately never unpacked/evaluated here — dev only touches train/valid.
    # (History sequences are built from all rows' timestamps, since a user's real watch
    # history is legitimately known at prediction time — that's not test-label leakage,
    # it's just not looking at test *labels/scores* during development.)
    video_field = FIELDS.index('video_id')

    model = SharedEmbeddingFM(dim + 1, tasks=[MAIN], k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(utr)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            b_idx = idx[i:i + bs]
            model.begin_batch()
            losses.append(pointwise_seq_contribution(
                model, MAIN, Xtr[b_idx], Htr[b_idx], Mtr[b_idx], ytr[b_idx], len(b_idx), video_field))
            model.commit()
        va = evaluate(uva, yva, predict_seq(model, MAIN, Xva, Hva, Mva, video_field))
        history.append(va)
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_state = va['primary'], 0, _snapshot(model)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    _restore_or_keep(best_state, model)
    return {'valid': evaluate(uva, yva, predict_seq(model, MAIN, Xva, Hva, Mva, video_field)),
            'epochs_run': len(history), 'history': history}


RUNNERS = {'A': run_A, 'B': run_B, 'C': run_C, 'D': run_D}
# baseline FM, valid split, per-seed (starter_kit/baseline.py --model fm --seed {0,1,2}) —
# reproduced locally; matched per-seed so deltas aren't confounded by seed variance.
BASELINE_VALID_BY_SEED = {
    0: {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015},
    1: {'GAUC': 0.6674, 'nDCG@5': 0.5361, 'primary': 0.6018},
    2: {'GAUC': 0.6671, 'nDCG@5': 0.5351, 'primary': 0.6011},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--iteration', default='all', choices=['A', 'B', 'C', 'D', 'all'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seeds', default='0', help='comma-separated seeds, e.g. 0,1,2')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    its = ['A', 'B', 'C', 'D'] if a.iteration == 'all' else [a.iteration]
    results = {}   # {it: {seed: res}}
    for it in its:
        results[it] = {}
        for seed in seeds:
            if seed not in BASELINE_VALID_BY_SEED:
                raise SystemExit(f"no reproduced baseline for seed={seed}; "
                                  f"run starter_kit/baseline.py --model fm --seed {seed} first")
            print(f"\n=== iteration {it} seed {seed} | hypothesis: {HYPOTHESES[it]} ===")
            t0 = time.time()
            res = RUNNERS[it](a.data_dir, k=a.k, lr=a.lr, epochs=a.epochs, seed=seed)
            elapsed = time.time() - t0
            res['wall_time_s'] = round(elapsed, 1)
            results[it][seed] = res
            v = res['valid']
            base = BASELINE_VALID_BY_SEED[seed]
            delta = v['primary'] - base['primary']
            print(f"  -> valid GAUC {v['GAUC']:.4f} | nDCG@5 {v['nDCG@5']:.4f} | primary {v['primary']:.4f} "
                  f"| delta vs baseline(seed{seed}) {delta:+.4f} | {elapsed:.1f}s | epochs {res['epochs_run']}")

    os.makedirs('logs', exist_ok=True)
    out_path = f"logs/iteration_{a.iteration}_seeds{'-'.join(map(str, seeds))}.json"
    with open(out_path, 'w') as fh:
        json.dump({
            'baseline_valid_by_seed': BASELINE_VALID_BY_SEED,
            'config': {'k': a.k, 'lr': a.lr, 'epochs': a.epochs, 'seeds': seeds},
            'hypotheses': HYPOTHESES,
            'results': {it: {str(seed): {'valid': r['valid'], 'epochs_run': r['epochs_run'],
                                          'wall_time_s': r['wall_time_s']}
                              for seed, r in seed_res.items()}
                        for it, seed_res in results.items()},
        }, fh, indent=2, default=float)
    print(f"\nlogged to {out_path}")

    print(f"\n{'iteration':10s} " + "".join(f"seed{s} d".rjust(10) for s in seeds) + f"{'mean d':>10s}")
    for it, seed_res in results.items():
        deltas = [seed_res[s]['valid']['primary'] - BASELINE_VALID_BY_SEED[s]['primary'] for s in seeds]
        print(f"{it:10s} " + "".join(f"{d:+10.4f}" for d in deltas) + f"{sum(deltas)/len(deltas):+10.4f}")


if __name__ == '__main__':
    main()
