"""Train MMoE: id embeddings (experiments/data_mt.py's vocab) + causal numeric
features (experiments/data_causal.py), jointly predicting long_view (primary) plus
auxiliary is_click/is_like/is_follow (experiments/mmoe_torch.py) through shared,
gated experts. Final ranking score is the long_view tower's own sigmoid output
(not multiplied with auxiliary-task probabilities — KuaiRand's tab/autoplay
impressions mean is_click doesn't cleanly gate long_view the way ESMM's
click->conversion funnel assumes, so an ESMM-style product risks injecting noise
rather than signal; the auxiliary tasks instead only shape the shared experts via
the joint loss, which is MTL's actual mechanism of benefit). Same per-user
weighted BCE recipe as DCN/DeepFM/BST for a fair comparison, applied per task.
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

from starter_kit.evaluate import evaluate
from experiments.data_mt import load_mt, encode_mt, AUX_TASKS
from experiments.data_causal import encode_causal
from experiments.mmoe_torch import MMoE, best_device
from experiments.train_dcn import build_numeric

BASELINE_VALID = {'GAUC': 0.6671, 'nDCG@5': 0.5358, 'primary': 0.6015}
BLEND_BEST = 0.6348
DCN_BEST = 0.6206
LABEL = 'long_view'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--emb_dim', type=int, default=16)
    ap.add_argument('--n_experts', type=int, default=4)
    ap.add_argument('--expert_dims', default='64,32')
    ap.add_argument('--tower_dims', default='16')
    ap.add_argument('--aux_weight', type=float, default=0.2)
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
    expert_dims = tuple(int(x) for x in a.expert_dims.split(','))
    tower_dims = tuple(int(x) for x in a.tower_dims.split(','))
    tasks = [LABEL] + list(AUX_TASKS)

    device = best_device()
    print(f"device = {device}")

    print("=== building features (id embeddings + causal numerics + aux labels) ===")
    t0 = time.time()
    splits = load_mt(a.data_dir)
    enc_mt, dim = encode_mt(splits)
    enc_causal, feature_names = encode_causal(a.data_dir)
    Xtr, Ytr, utr = enc_mt['train']
    Xva, Yva, uva = enc_mt['valid']
    assert utr == enc_causal['train']['users'] and uva == enc_causal['valid']['users'], \
        "row order mismatch between data_mt and data_causal"
    num_tr, mean, std = build_numeric(enc_causal, 'train')
    num_va, _, _ = build_numeric(enc_causal, 'valid', mean, std)
    yva_primary = enc_causal['valid']['y']
    print(f"  ready in {time.time()-t0:.1f}s, {num_tr.shape[1]} numeric dims, "
          f"tasks={tasks}, aux_weight={a.aux_weight}")
    for t in tasks:
        print(f"  train {t} positive rate: {Ytr[t].mean():.4f}")

    group_map = dict(zip(*np.unique(np.array(utr), return_counts=True)))
    w_tr = np.array([1.0 / group_map[u] for u in utr], dtype=np.float32)

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.long, device=device)
    num_tr_t = torch.as_tensor(num_tr, dtype=torch.float32, device=device)
    wtr_t = torch.as_tensor(w_tr, dtype=torch.float32, device=device)
    Ytr_t = {t: torch.as_tensor(Ytr[t], dtype=torch.float32, device=device) for t in tasks}
    Xva_t = torch.as_tensor(Xva, dtype=torch.long, device=device)
    num_va_t = torch.as_tensor(num_va, dtype=torch.float32, device=device)

    torch.manual_seed(a.seed)
    model = MMoE(dim, num_fields=Xtr.shape[1], num_numeric=num_tr.shape[1], tasks=tasks,
                 emb_dim=a.emb_dim, n_experts=a.n_experts, expert_dims=expert_dims,
                 tower_dims=tower_dims, dropout=a.dropout, seed=a.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    n = Xtr_t.shape[0]
    gen = torch.Generator(device=device).manual_seed(a.seed)
    task_weight = {LABEL: 1.0, **{t: a.aux_weight for t in AUX_TASKS}}

    best, best_snap, bad = -1.0, None, 0
    print(f"\n=== training MMoE ===")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        losses = []
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr_t[idx], num_tr_t[idx])
            loss = 0.0
            for t in tasks:
                loss = loss + task_weight[t] * F.binary_cross_entropy_with_logits(
                    logits[t], Ytr_t[t][idx], weight=wtr_t[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            zva = model(Xva_t, num_va_t)[LABEL]
            pva = torch.sigmoid(zva).cpu().numpy()
        m = evaluate(uva, yva_primary, pva)
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
        pva = torch.sigmoid(model(Xva_t, num_va_t)[LABEL]).cpu().numpy()
    final = evaluate(uva, yva_primary, pva)
    print(f"\n=== MMoE valid: GAUC {final['GAUC']:.4f} nDCG@5 {final['nDCG@5']:.4f} "
          f"primary {final['primary']:.4f} ===")
    print(f"delta vs baseline: {final['primary']-BASELINE_VALID['primary']:+.4f}")
    print(f"delta vs DCN ({DCN_BEST}): {final['primary']-DCN_BEST:+.4f}")
    print(f"delta vs blend best ({BLEND_BEST}): {final['primary']-BLEND_BEST:+.4f}")

    np.save(os.path.join(a.out_dir, 'mmoe_valid_scores.npy'), pva)
    np.save(os.path.join(a.out_dir, 'mmoe_valid_users.npy'), np.array(uva, dtype=object))
    np.save(os.path.join(a.out_dir, 'mmoe_valid_labels.npy'), yva_primary)
    with open('logs/iteration_Z_mmoe.json', 'w') as fh:
        json.dump({'baseline_valid': BASELINE_VALID, 'dcn_best': DCN_BEST, 'blend_best': BLEND_BEST,
                    'config': vars(a), 'valid': final}, fh, indent=2, default=float)
    print("logged to logs/iteration_Z_mmoe.json")


if __name__ == '__main__':
    main()
