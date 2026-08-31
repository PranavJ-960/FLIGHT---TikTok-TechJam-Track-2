"""Train BST (Behavior Sequence Transformer): a genuine self-attention sequence
model over each user's real chronological history (experiments/data_seq.py,
max_len=20 — a true multi-step Transformer, unlike DCN/DeepFM's single-step
DIN-style attention or iteration D's earlier attempt), target video appended as
the final sequence token, output concatenated with the proven causal numeric
features. Same per-user weighted BCE recipe as DCN/DeepFM for a fair comparison.
Train/valid only; test never touched.
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

from starter_kit.data import FIELDS
from starter_kit.evaluate import evaluate
from experiments.data_seq import encode_seq
from experiments.data_causal import encode_causal
from experiments.bst_torch import BST, best_device
from experiments.train_dcn import build_numeric

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
BLEND_BEST = 0.6348
DCN_BEST = 0.6206


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--emb_dim', type=int, default=16)
    ap.add_argument('--n_heads', type=int, default=2)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--mlp_dims', default='64,32')
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    mlp_dims = tuple(int(x) for x in a.mlp_dims.split(','))

    device = best_device()
    print(f"device = {device}")

    print("=== building features (history sequences + causal numerics) ===")
    t0 = time.time()
    enc_seq, dim, pad_idx = encode_seq(a.data_dir, max_len=a.max_len)
    enc_causal, feature_names = encode_causal(a.data_dir)
    Xtr, Htr, Mtr, ytr_seq, utr = enc_seq['train']
    Xva, Hva, Mva, yva_seq, uva = enc_seq['valid']
    assert utr == enc_causal['train']['users'] and uva == enc_causal['valid']['users'], \
        "row order mismatch between data_seq and data_causal"
    num_tr, mean, std = build_numeric(enc_causal, 'train')
    num_va, _, _ = build_numeric(enc_causal, 'valid', mean, std)
    ytr = enc_causal['train']['y']
    yva = enc_causal['valid']['y']
    video_field = FIELDS.index('video_id')
    print(f"  ready in {time.time()-t0:.1f}s, {num_tr.shape[1]} numeric dims, max_len={a.max_len}")

    group_map = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr], dtype=np.float32)

    tgt_tr_t = torch.as_tensor(Xtr[:, video_field], dtype=torch.long, device=device)
    Htr_t = torch.as_tensor(Htr, dtype=torch.long, device=device)
    Mtr_t = torch.as_tensor(Mtr, dtype=torch.float32, device=device)
    num_tr_t = torch.as_tensor(num_tr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    wtr_t = torch.as_tensor(w_tr, dtype=torch.float32, device=device)

    tgt_va_t = torch.as_tensor(Xva[:, video_field], dtype=torch.long, device=device)
    Hva_t = torch.as_tensor(Hva, dtype=torch.long, device=device)
    Mva_t = torch.as_tensor(Mva, dtype=torch.float32, device=device)
    num_va_t = torch.as_tensor(num_va, dtype=torch.float32, device=device)

    torch.manual_seed(a.seed)
    model = BST(dim, a.max_len, num_tr.shape[1], emb_dim=a.emb_dim, n_heads=a.n_heads,
                n_layers=a.n_layers, mlp_dims=mlp_dims, dropout=a.dropout, seed=a.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    n = tgt_tr_t.shape[0]
    gen = torch.Generator(device=device).manual_seed(a.seed)

    best, best_snap, bad = -1.0, None, 0
    print(f"\n=== training BST ===")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        losses = []
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            opt.zero_grad(set_to_none=True)
            z = model(tgt_tr_t[idx], Htr_t[idx], Mtr_t[idx], num_tr_t[idx])
            loss = F.binary_cross_entropy_with_logits(z, ytr_t[idx], weight=wtr_t[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            zva = model(tgt_va_t, Hva_t, Mva_t, num_va_t)
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
        pva = torch.sigmoid(model(tgt_va_t, Hva_t, Mva_t, num_va_t)).cpu().numpy()
    final = evaluate(uva, yva, pva)
    print(f"\n=== BST valid: GAUC {final['GAUC']:.4f} nDCG@5 {final['nDCG@5']:.4f} "
          f"primary {final['primary']:.4f} ===")
    print(f"delta vs baseline: {final['primary']-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs DCN ({DCN_BEST}): {final['primary']-DCN_BEST:+.4f}")
    print(f"delta vs blend best ({BLEND_BEST}): {final['primary']-BLEND_BEST:+.4f}")

    np.save(os.path.join(a.out_dir, 'bst_valid_scores.npy'), pva)
    np.save(os.path.join(a.out_dir, 'bst_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'bst_valid_labels.npy'), yva)
    with open('logs/iteration_Y_bst.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'dcn_best': DCN_BEST, 'blend_best': BLEND_BEST,
                    'config': vars(a), 'valid': final}, fh, indent=2, default=float)
    print("logged to logs/iteration_Y_bst.json")


if __name__ == '__main__':
    main()
