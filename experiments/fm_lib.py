"""Shared FM model + loss building blocks for iterations A/B/C/D.

Generalizes starter_kit/baseline.py's FM to (a) a shared embedding table V used
by several named tasks, each with its own linear weights/bias, (b) a pairwise
(BPR-style) loss alongside the original pointwise BCE loss, and (c) a DIN-style
attention term over a user's history sequence, folded into the FM interaction
as one extra "virtual field". Passing a single task with only pointwise steps
and no history reproduces the baseline FM exactly. starter_kit/ is not modified.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SharedEmbeddingFM:
    def __init__(self, dim, tasks, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.tasks = list(tasks)
        self.W = {t: np.zeros(dim, dtype=np.float32) for t in self.tasks}
        self.b = {t: np.float32(0.0) for t in self.tasks}
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = {t: np.zeros(dim, dtype=np.float32) for t in self.tasks}
        self.vW = {t: np.zeros(dim, dtype=np.float32) for t in self.tasks}
        self.t = 0

    def interaction(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                     # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return inter, E, S

    def logits(self, X, task):
        inter, E, S = self.interaction(X)
        z = self.b[task] + self.W[task][X].sum(1) + inter
        return z, E, S

    def _adam(self, P, G, M, Vv):
        b1, b2, eps = 0.9, 0.999, 1e-8
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def begin_batch(self):
        self.t += 1
        self._gV = np.zeros_like(self.V)
        self._gW = {t: np.zeros_like(self.W[t]) for t in self.tasks}
        self._gb = {t: 0.0 for t in self.tasks}

    def add_contribution(self, task, X, g, E, S):
        np.add.at(self._gW[task], X, g[:, None])
        np.add.at(self._gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._gb[task] += float(g.sum())

    def commit(self):
        gV = self._gV + self.l2 * self.V
        self._adam(self.V, gV, self.mV, self.vV)
        for task in self.tasks:
            gW = self._gW[task] + self.l2 * self.W[task]
            self._adam(self.W[task], gW, self.mW[task], self.vW[task])
            self.b[task] -= self.lr * self._gb[task]

    def predict(self, X, task, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], task)[0] for i in range(0, len(X), bs)])


def pointwise_contribution(model, task, X, y, batch_size):
    z, E, S = model.logits(X, task)
    g = ((sigmoid(z) - y) / batch_size).astype(np.float32)
    model.add_contribution(task, X, g, E, S)
    return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))


def pairwise_contribution(model, task, X_pos, X_neg, batch_size):
    z_pos, E_pos, S_pos = model.logits(X_pos, task)
    z_neg, E_neg, S_neg = model.logits(X_neg, task)
    diff = z_pos - z_neg
    g = ((sigmoid(diff) - 1.0) / batch_size).astype(np.float32)
    model.add_contribution(task, X_pos, g, E_pos, S_pos)
    model.add_contribution(task, X_neg, -g, E_neg, S_neg)
    return float(-np.mean(np.log(sigmoid(diff) + 1e-9)))


def build_user_pos_neg(y, users):
    """One-time grouping of train row indices into per-user pos/neg index arrays."""
    by_user = {}
    for i, (u, yy) in enumerate(zip(users, y)):
        pos, neg = by_user.setdefault(u, ([], []))
        (pos if yy > 0 else neg).append(i)
    groups = {}
    for u, (pos, neg) in by_user.items():
        if pos and neg:
            groups[u] = (np.asarray(neg, dtype=np.int64), np.asarray(pos, dtype=np.int64))
    return groups


def logits_with_history(model, task, X, H, M, video_field):
    """Forward pass: FM logit for X, plus a DIN-style attention pool over history
    H/M, folded in as one extra interaction-only field u (no linear weight, since
    u has no discrete id to look one up with).
    """
    k = model.V.shape[1]
    E = model.V[X]                                          # (B,F,k)
    v_cand = E[:, video_field, :]                            # (B,k)
    v_hist = model.V[H]                                      # (B,L,k)
    scores = (v_hist * v_cand[:, None, :]).sum(-1) / np.sqrt(k)
    scores = np.where(M > 0, scores, -1e9)
    scores = scores - scores.max(axis=1, keepdims=True)
    ex = np.exp(scores) * M
    denom = ex.sum(1, keepdims=True)
    alpha = np.divide(ex, denom, out=np.zeros_like(ex), where=denom > 0)
    u = (alpha[:, :, None] * v_hist).sum(1)                  # (B,k)

    E_ext = np.concatenate([E, u[:, None, :]], axis=1)       # (B,F+1,k)
    S_ext = E_ext.sum(1)
    inter = 0.5 * ((S_ext ** 2).sum(1) - (E_ext ** 2).sum((1, 2)))
    z = model.b[task] + model.W[task][X].sum(1) + inter
    cache = (E, v_cand, v_hist, alpha, u, S_ext, k)
    return z, cache


def add_history_contribution(model, task, X, H, M, g, cache, video_field):
    E, v_cand, v_hist, alpha, u, S_ext, k = cache

    np.add.at(model._gW[task], X, g[:, None])
    model._gb[task] += float(g.sum())

    diff_disc = S_ext[:, None, :] - E                        # (B,F,k), per discrete field
    np.add.at(model._gV, X, g[:, None, None] * diff_disc)

    gu = g[:, None] * (S_ext - u)                             # (B,k) dL/du

    dL_dalpha = (gu[:, None, :] * v_hist).sum(-1)              # (B,L)
    c = (alpha * dL_dalpha).sum(1, keepdims=True)
    dL_ds = alpha * (dL_dalpha - c) / np.sqrt(k)               # softmax backward, already 0 where alpha==0

    grad_v_hist = alpha[:, :, None] * gu[:, None, :] + dL_ds[:, :, None] * v_cand[:, None, :]
    np.add.at(model._gV, H, grad_v_hist)

    grad_v_cand_attn = (dL_ds[:, :, None] * v_hist).sum(1)
    np.add.at(model._gV, X[:, video_field], grad_v_cand_attn)


def pointwise_seq_contribution(model, task, X, H, M, y, batch_size, video_field):
    z, cache = logits_with_history(model, task, X, H, M, video_field)
    g = ((sigmoid(z) - y) / batch_size).astype(np.float32)
    add_history_contribution(model, task, X, H, M, g, cache, video_field)
    return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))


def predict_seq(model, task, X, H, M, video_field, bs=200_000):
    out = []
    for i in range(0, len(X), bs):
        z, _ = logits_with_history(model, task, X[i:i + bs], H[i:i + bs], M[i:i + bs], video_field)
        out.append(z)
    return np.concatenate(out)


def sample_pairs(groups, rng):
    """Resample one (pos,neg) pool for one epoch. Pool size per user ~= max(#pos,#neg)
    so active users contribute roughly as many pairs as they have impressions."""
    pos_chunks, neg_chunks = [], []
    for neg, pos in groups.values():
        n = max(len(pos), len(neg))
        pos_chunks.append(rng.choice(pos, size=n, replace=True))
        neg_chunks.append(rng.choice(neg, size=n, replace=True))
    pos_idx = np.concatenate(pos_chunks)
    neg_idx = np.concatenate(neg_chunks)
    perm = rng.permutation(len(pos_idx))
    return pos_idx[perm], neg_idx[perm]
