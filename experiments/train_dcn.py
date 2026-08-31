"""Train the DCN (Deep & Cross Network) model: id embeddings (from experiments/
data_seq.py's vocab, same as FM) + the proven causal engineered features (from
experiments/data_causal.py) as numeric inputs, combined via an explicit cross network
and a deep MLP. Trained with per-user weighted BCE (weight=1/user_group_size) — the
single biggest lever found for the GBT line of work, applied here too. Train/valid
only; test never touched. Standalone process (no lightgbm/xgboost/catboost import —
avoids the established torch/lightgbm native-threading deadlock).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from starter_kit.evaluate import evaluate
from experiments.data_seq import encode_seq
from experiments.data_causal import encode_causal, NUMERIC_FIELDS
from experiments.dcn_torch import DCN, best_device

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LINEAR_BLEND_BEST = 0.6359
NAN_COLS = ['user_since_ms', 'video_since_ms', 'author_since_ms']


def build_numeric(enc_causal, split, mean=None, std=None):
    d = enc_causal[split]
    num = d['num'].copy()                       # (N, 16)
    nan_idx = [NUMERIC_FIELDS.index(c) for c in NAN_COLS]
    missing = np.isnan(num[:, nan_idx]).astype(np.float32)          # (N, 3)
    num = np.nan_to_num(num, nan=0.0)
    if mean is None:
        mean = num.mean(axis=0, keepdims=True)
        std = num.std(axis=0, keepdims=True) + 1e-6
    num_z = (num - mean) / std
    dow_onehot = np.eye(7, dtype=np.float32)[d['dow']]
    full = np.concatenate([num_z, missing, dow_onehot], axis=1).astype(np.float32)
    return full, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--emb_dim', type=int, default=16)
    ap.add_argument('--cross_layers', type=int, default=3)
    ap.add_argument('--deep_dims', default='128,64')
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    deep_dims = tuple(int(x) for x in a.deep_dims.split(','))

    device = best_device()
    print(f"device = {device}")

    print("=== building features (id embeddings + causal numerics) ===")
    t0 = time.time()
    enc_seq, dim, pad_idx = encode_seq(a.data_dir, max_len=1)   # max_len=1: history unused here
    enc_causal, feature_names = encode_causal(a.data_dir)
    Xtr_seq, _, _, ytr_seq, utr = enc_seq['train']
    Xva_seq, _, _, yva_seq, uva = enc_seq['valid']
    assert utr == enc_causal['train']['users'] and uva == enc_causal['valid']['users'], \
        "row order mismatch between data_seq and data_causal"
    num_tr, mean, std = build_numeric(enc_causal, 'train')
    num_va, _, _ = build_numeric(enc_causal, 'valid', mean, std)
    ytr = enc_causal['train']['y']
    yva = enc_causal['valid']['y']
    print(f"  ready in {time.time()-t0:.1f}s, {num_tr.shape[1]} numeric dims, 5 embedded fields")

    group_map = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr], dtype=np.float32)

    Xtr_t = torch.as_tensor(Xtr_seq, dtype=torch.long, device=device)
    num_tr_t = torch.as_tensor(num_tr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    wtr_t = torch.as_tensor(w_tr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva_seq, dtype=torch.long, device=device)
    num_va_t = torch.as_tensor(num_va, dtype=torch.float32, device=device)

    torch.manual_seed(a.seed)
    model = DCN(dim, num_fields=5, num_numeric=num_tr.shape[1], emb_dim=a.emb_dim,
                cross_layers=a.cross_layers, deep_dims=deep_dims, dropout=a.dropout, seed=a.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    n = Xtr_t.shape[0]
    gen = torch.Generator(device=device).manual_seed(a.seed)

    best, best_snap, bad = -1.0, None, 0
    print(f"\n=== training DCN ===")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        losses = []
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            opt.zero_grad(set_to_none=True)
            z = model(Xtr_t[idx], num_tr_t[idx])
            loss = F.binary_cross_entropy_with_logits(z, ytr_t[idx], weight=wtr_t[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            zva = model(Xva_t, num_va_t)
            pva = torch.sigmoid(zva).cpu().numpy()
        m = evaluate(uva, yva, pva)
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {m['primary']:.4f} "
              f"| {time.time()-t0:.1f}s")
        if m['primary'] > best + 1e-5:
            best, bad, best_snap = m['primary'], 0, model.state_snapshot()
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep}")
                break

    model.load_snapshot(best_snap)
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva_t, num_va_t)).cpu().numpy()
    final = evaluate(uva, yva, pva)
    print(f"\n=== DCN valid: GAUC {final['GAUC']:.4f} nDCG@5 {final['nDCG@5']:.4f} "
          f"primary {final['primary']:.4f} ===")
    print(f"delta vs baseline: {final['primary']-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs linear blend best ({LINEAR_BLEND_BEST}): {final['primary']-LINEAR_BLEND_BEST:+.4f}")

    np.save(os.path.join(a.out_dir, 'dcn_valid_scores.npy'), pva)
    np.save(os.path.join(a.out_dir, 'dcn_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'dcn_valid_labels.npy'), yva)
    with open('logs/iteration_S_dcn.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'linear_blend_best': LINEAR_BLEND_BEST,
                    'config': vars(a), 'valid': final}, fh, indent=2, default=float)
    print("logged to logs/iteration_S_dcn.json")


if __name__ == '__main__':
    main()
