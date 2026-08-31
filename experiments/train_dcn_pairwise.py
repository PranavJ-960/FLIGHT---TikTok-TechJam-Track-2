"""DCN trained with a pairwise (BPR) ranking loss instead of pointwise BCE — the
metric (GAUC/nDCG@5) is a within-user ranking metric, so directly optimizing
"does the model rank this user's positive above their negatives" is a more direct
match than binary cross-entropy on each row independently (the same reasoning
lambdarank/rank:ndcg/YetiRank already apply to the GBT side). Earlier hypothesis A
tested BPR on plain FM (id embeddings only, no causal features) and landed at
roughly baseline; this reuses the same causal features + id embeddings as the
best neural net tried (DCN, iteration S) so pairwise loss gets a fair shot on the
strongest available backbone, not the weakest. For each user with at least one
positive and one negative impression, sample (positive, negative) pairs and train
with the BPR loss -log(sigmoid(score_pos - score_neg)), per-user pair-weighted
(1/pairs_for_that_user) for the same reason per-user row weighting helped BCE
elsewhere: GAUC/nDCG@5 average per user, so a power user's pairs shouldn't dominate
the gradient just because they generate more of them. Train/valid only; test never
touched.
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
from experiments.data_causal import encode_causal
from experiments.dcn_torch import DCN, best_device
from experiments.train_dcn import build_numeric

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
LINEAR_BLEND_BEST = 0.6348
DCN_BCE_BEST = 0.6206


def build_pairs(y, users, rng, max_pairs_per_user=20):
    """For each user's impression group, pair every positive with up to
    max_pairs_per_user random negatives (or all negatives if fewer). Returns
    (pos_idx, neg_idx, pair_user_weight) arrays over the full training set."""
    order = np.argsort(users, kind='stable')
    users_s = np.asarray(users)[order]
    y_s = y[order]
    _, starts, counts = np.unique(users_s, return_index=True, return_counts=True)

    pos_list, neg_list, w_list = [], [], []
    for start, count in zip(starts, counts):
        idx = order[start:start + count]
        pos = idx[y_s[start:start + count] == 1]
        neg = idx[y_s[start:start + count] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pairs = min(max_pairs_per_user, len(pos) * len(neg))
        p = rng.choice(pos, size=n_pairs, replace=True)
        n = rng.choice(neg, size=n_pairs, replace=True)
        pos_list.append(p)
        neg_list.append(n)
        w_list.append(np.full(n_pairs, 1.0 / n_pairs, dtype=np.float32))
    return (np.concatenate(pos_list), np.concatenate(neg_list), np.concatenate(w_list))


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
    ap.add_argument('--max_pairs_per_user', type=int, default=20)
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
    enc_seq, dim, pad_idx = encode_seq(a.data_dir, max_len=1)
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

    rng = np.random.default_rng(a.seed)
    pos_idx, neg_idx, pair_w = build_pairs(ytr, utr, rng, a.max_pairs_per_user)
    print(f"  built {len(pos_idx)} training pairs from {len(np.unique(utr))} users")

    Xtr_t = torch.as_tensor(Xtr_seq, dtype=torch.long, device=device)
    num_tr_t = torch.as_tensor(num_tr, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva_seq, dtype=torch.long, device=device)
    num_va_t = torch.as_tensor(num_va, dtype=torch.float32, device=device)
    pos_t = torch.as_tensor(pos_idx, dtype=torch.long, device=device)
    neg_t = torch.as_tensor(neg_idx, dtype=torch.long, device=device)
    w_t = torch.as_tensor(pair_w, dtype=torch.float32, device=device)

    torch.manual_seed(a.seed)
    model = DCN(dim, num_fields=5, num_numeric=num_tr.shape[1], emb_dim=a.emb_dim,
                cross_layers=a.cross_layers, deep_dims=deep_dims, dropout=a.dropout, seed=a.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    n_pairs = pos_t.shape[0]
    gen = torch.Generator(device=device).manual_seed(a.seed)

    best, best_snap, bad = -1.0, None, 0
    print(f"\n=== training DCN with pairwise (BPR) loss ===")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_pairs, device=device, generator=gen)
        losses = []
        for i in range(0, n_pairs, a.bs):
            idx = perm[i:i + a.bs]
            p_idx, n_idx, w = pos_t[idx], neg_t[idx], w_t[idx]
            opt.zero_grad(set_to_none=True)
            z_pos = model(Xtr_t[p_idx], num_tr_t[p_idx])
            z_neg = model(Xtr_t[n_idx], num_tr_t[n_idx])
            per_pair_loss = -F.logsigmoid(z_pos - z_neg)
            loss = (per_pair_loss * w).sum() / w.sum()
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
    print(f"\n=== DCN (pairwise BPR) valid: GAUC {final['GAUC']:.4f} nDCG@5 {final['nDCG@5']:.4f} "
          f"primary {final['primary']:.4f} ===")
    print(f"delta vs baseline: {final['primary']-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs DCN-BCE ({DCN_BCE_BEST}): {final['primary']-DCN_BCE_BEST:+.4f}")
    print(f"delta vs blend best ({LINEAR_BLEND_BEST}): {final['primary']-LINEAR_BLEND_BEST:+.4f}")

    np.save(os.path.join(a.out_dir, 'dcn_pairwise_valid_scores.npy'), pva)
    np.save(os.path.join(a.out_dir, 'dcn_pairwise_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'dcn_pairwise_valid_labels.npy'), yva)
    with open('logs/iteration_X_dcn_pairwise.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'dcn_bce_best': DCN_BCE_BEST,
                    'blend_best': LINEAR_BLEND_BEST, 'config': vars(a), 'valid': final,
                    'n_pairs': int(n_pairs)}, fh, indent=2, default=float)
    print("logged to logs/iteration_X_dcn_pairwise.json")


if __name__ == '__main__':
    main()
