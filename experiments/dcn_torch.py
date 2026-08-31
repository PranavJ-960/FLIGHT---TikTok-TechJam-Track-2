"""DCN-style (Deep & Cross Network) model: a genuinely different architecture from
both FM (iterations A-D, bilinear only) and the GBT ensemble (LightGBM/XGBoost/
CatBoost) — combines learned id embeddings (like FM) with the proven causal features
(like the GBTs) through an explicit polynomial cross network plus a deep MLP tower,
in one jointly-trained model. Uses the same shared-embedding-table convention as
experiments/fm_torch.py (V indexed by already-offset field ids).
"""
import torch
import torch.nn as nn


def best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class DCN(nn.Module):
    def __init__(self, vocab_dim, num_fields, num_numeric, emb_dim=16, cross_layers=3,
                 deep_dims=(128, 64), dropout=0.2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(vocab_dim, emb_dim, generator=g) * 0.01)
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        input_dim = num_fields * emb_dim + num_numeric
        self.input_dim = input_dim

        self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(input_dim, generator=g) * 0.01)
                                          for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)])

        deep_layers = []
        prev = input_dim
        for h in deep_dims:
            deep_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep_layers)
        self.out = nn.Linear(input_dim + deep_dims[-1], 1)

    def forward(self, X, num_x):
        emb = self.V[X]                                   # (B, F, k)
        e_flat = emb.reshape(emb.shape[0], -1)             # (B, F*k)
        x0 = torch.cat([e_flat, num_x], dim=1)             # (B, input_dim)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xw = (xl * w).sum(dim=1, keepdim=True)          # (B,1)
            xl = x0 * xw + b + xl
        deep_out = self.deep(x0)
        combined = torch.cat([xl, deep_out], dim=1)
        return self.out(combined).squeeze(-1)

    def state_snapshot(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_snapshot(self, snap):
        self.load_state_dict(snap)
