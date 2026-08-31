"""GPU (MPS) training loop for the DIN-style FM (iteration D), using fm_torch's
autograd model. Train/valid only — test is never unpacked or evaluated here,
matching experiments/run_iteration.py's rule.
"""
import argparse
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
from experiments.fm_torch import TorchFM, best_device

MAIN = 'long_view'
VIDEO_FIELD = FIELDS.index('video_id')


def to_device_tensors(enc, split, device):
    X, H, M, y, users = enc[split]
    return (torch.as_tensor(X, dtype=torch.long, device=device),
            torch.as_tensor(H, dtype=torch.long, device=device),
            torch.as_tensor(M, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device),
            y, users)   # keep numpy y + users too, evaluate() wants those


def train_D_torch(tensors_tr, tensors_va, dim, k=16, lr=0.001, l2=1e-6, bs=8192,
                   epochs=40, patience=4, seed=0, device=None, verbose=True, return_model=False):
    device = device or best_device()
    Xtr, Htr, Mtr, ytr, _, _ = tensors_tr
    Xva, Hva, Mva, _, yva_np, uva = tensors_va

    torch.manual_seed(seed)
    model = TorchFM(dim + 1, tasks=[MAIN], k=k, seed=seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    n = Xtr.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)

    best, best_snap, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        losses = []
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            z = model.logits_with_history(Xtr[idx], Htr[idx], Mtr[idx], MAIN, VIDEO_FIELD)
            loss = F.binary_cross_entropy_with_logits(z, ytr[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            zva = model.logits_with_history(Xva, Hva, Mva, MAIN, VIDEO_FIELD)
            pva = torch.sigmoid(zva).cpu().numpy()
        va = evaluate(uva, yva_np, pva)
        if verbose:
            print(f"    epoch {ep:2d} | loss {np.mean(losses):.4f} | valid primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.2f}s")
        if va['primary'] > best + 1e-5:
            best, bad, best_snap = va['primary'], 0, model.state_snapshot()
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep}")
                break
    model.load_snapshot(best_snap)
    model.eval()
    with torch.no_grad():
        zva = model.logits_with_history(Xva, Hva, Mva, MAIN, VIDEO_FIELD)
        pva = torch.sigmoid(zva).cpu().numpy()
    metrics = evaluate(uva, yva_np, pva)
    return (metrics, pva, model) if return_model else (metrics, pva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--max_len', type=int, default=20)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()

    device = torch.device(a.device) if a.device else best_device()
    print(f"device = {device}")

    print("loading sequence-encoded data ...")
    t0 = time.time()
    enc, dim, pad_idx = encode_seq(a.data_dir, max_len=a.max_len)
    print(f"  loaded in {time.time()-t0:.1f}s")

    tensors_tr = to_device_tensors(enc, 'train', device)
    tensors_va = to_device_tensors(enc, 'valid', device)

    print(f"\n=== D (torch/{device.type}) k={a.k} lr={a.lr} seed={a.seed} ===")
    t0 = time.time()
    va, _ = train_D_torch(tensors_tr, tensors_va, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                           seed=a.seed, device=device, verbose=True)
    print(f"-> valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
          f"| total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
