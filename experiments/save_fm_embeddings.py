"""Train FM once (k=16, lr=0.0006 — the tuned config) and save its learned user_id/
video_id embedding rows per train/valid row, for use as LightGBM input features
(iteration L: representation-level stacking, unlike blend_scores.py's prediction-level
blending which didn't help). Standalone process — kept separate from lightgbm (see
experiments/train_lgb.py's docstring on the torch/lightgbm deadlock). Train/valid only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from starter_kit.data import FIELDS
from experiments.data_seq import encode_seq
from experiments.train_torch import to_device_tensors, train_D_torch
from experiments.fm_torch import best_device

BEST_CONFIG = {'k': 16, 'lr': 0.0006}
USER_FIELD = FIELDS.index('user_id')
VIDEO_FIELD = FIELDS.index('video_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    device = best_device()
    print(f"device = {device}")
    enc, dim, pad_idx = encode_seq(a.data_dir, max_len=20)
    tensors_tr = to_device_tensors(enc, 'train', device)
    tensors_va = to_device_tensors(enc, 'valid', device)

    va, _, model = train_D_torch(tensors_tr, tensors_va, dim, k=BEST_CONFIG['k'], lr=BEST_CONFIG['lr'],
                                  seed=a.seed, device=device, verbose=False, return_model=True)
    print(f"FM trained, valid primary {va['primary']:.4f}")

    V = model.V.detach().cpu().numpy()   # (dim+1, k)
    Xtr = tensors_tr[0].cpu().numpy()
    Xva = tensors_va[0].cpu().numpy()
    users_tr = enc['train'][4]
    users_va = enc['valid'][4]

    for split, X, users in (('train', Xtr, users_tr), ('valid', Xva, users_va)):
        user_emb = V[X[:, USER_FIELD]]
        video_emb = V[X[:, VIDEO_FIELD]]
        np.save(os.path.join(a.out_dir, f'fm_user_emb_{split}.npy'), user_emb.astype(np.float32))
        np.save(os.path.join(a.out_dir, f'fm_video_emb_{split}.npy'), video_emb.astype(np.float32))
        np.save(os.path.join(a.out_dir, f'fm_emb_{split}_users.npy'), np.array(users, dtype=object))
        print(f"  {split}: user_emb {user_emb.shape}, video_emb {video_emb.shape}")

    print("saved FM embeddings to", a.out_dir)


if __name__ == '__main__':
    main()
