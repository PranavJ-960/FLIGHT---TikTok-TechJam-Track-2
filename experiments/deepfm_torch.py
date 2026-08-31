"""DeepFM: a genuinely different architecture from DCN (iteration S, cross network)
and from FM-alone (iterations A-D) — combines an explicit FM second-order interaction
term over the id embeddings (same shared-embedding-table convention as
experiments/fm_torch.py / dcn_torch.py) with a deep MLP tower over the same
embeddings plus the causal numeric features, summed at the output (standard
DeepFM: y = linear + FM_2nd_order(sparse) + MLP(sparse ++ dense)). Numeric
features get their own linear term (dense features aren't run through the FM
interaction — the common practical DeepFM variant, since a per-numeric-field
embedding for continuous features adds complexity without a clear benefit here).
"""
import torch
import torch.nn as nn


def best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class DeepFM(nn.Module):
    def __init__(self, vocab_dim, num_fields, num_numeric, emb_dim=16, deep_dims=(128, 64),
                 dropout=0.2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(vocab_dim, emb_dim, generator=g) * 0.01)
        self.field_bias = nn.Parameter(torch.zeros(vocab_dim))
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.numeric_linear = nn.Linear(num_numeric, 1)

        input_dim = num_fields * emb_dim + num_numeric
        deep_layers = []
        prev = input_dim
        for h in deep_dims:
            deep_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        deep_layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*deep_layers)

    def forward(self, X, num_x):
        emb = self.V[X]                                        # (B, F, k)
        linear_term = self.field_bias[X].sum(dim=1) + self.numeric_linear(num_x).squeeze(-1)

        sum_sq = emb.sum(dim=1) ** 2                            # (B, k)
        sq_sum = (emb ** 2).sum(dim=1)                          # (B, k)
        fm_2nd = 0.5 * (sum_sq - sq_sum).sum(dim=1)              # (B,)

        e_flat = emb.reshape(emb.shape[0], -1)                  # (B, F*k)
        x0 = torch.cat([e_flat, num_x], dim=1)
        deep_out = self.deep(x0).squeeze(-1)                    # (B,)

        return linear_term + fm_2nd + deep_out

    def state_snapshot(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_snapshot(self, snap):
        self.load_state_dict(snap)
