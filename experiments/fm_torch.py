"""PyTorch/MPS version of the shared-embedding FM + DIN-style attention model.
Same math as experiments/fm_lib.py (structurally identical forward pass — one
extra "virtual field" u from attention folded into the FM interaction term),
but using autograd instead of hand-rolled backprop, so it runs on GPU (Apple
Silicon MPS, or CUDA if present) and needs no manual gradient derivation.
starter_kit/ is not modified; this is purely an alternate compute backend for
the same model family already validated (gradient-checked) in fm_lib.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class TorchFM(nn.Module):
    def __init__(self, dim, tasks, k=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(dim, k, generator=g) * 0.01)
        self.tasks = list(tasks)
        self.W = nn.ParameterDict({t: nn.Parameter(torch.zeros(dim)) for t in self.tasks})
        self.bias = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in self.tasks})
        self.k = k

    def logits(self, X, task):
        E = self.V[X]                                    # (B,F,k)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.bias[task] + self.W[task][X].sum(1) + inter

    def logits_with_history(self, X, H, M, task, video_field):
        E = self.V[X]                                     # (B,F,k)
        v_cand = E[:, video_field, :]                      # (B,k)
        v_hist = self.V[H]                                 # (B,L,k)
        scores = (v_hist * v_cand.unsqueeze(1)).sum(-1) / (self.k ** 0.5)
        scores = scores.masked_fill(M <= 0, -1e9)
        scores = scores - scores.max(dim=1, keepdim=True).values
        ex = torch.exp(scores) * M
        denom = ex.sum(1, keepdim=True)
        alpha = ex / denom.clamp_min(1e-12)
        u = (alpha.unsqueeze(-1) * v_hist).sum(1)           # (B,k)

        E_ext = torch.cat([E, u.unsqueeze(1)], dim=1)       # (B,F+1,k)
        S_ext = E_ext.sum(1)
        inter = 0.5 * ((S_ext ** 2).sum(1) - (E_ext ** 2).sum((1, 2)))
        return self.bias[task] + self.W[task][X].sum(1) + inter

    def state_snapshot(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_snapshot(self, snap):
        self.load_state_dict(snap)
