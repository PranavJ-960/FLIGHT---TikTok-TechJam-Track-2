"""MMoE (Multi-gate Mixture-of-Experts): a genuine gated multi-task architecture,
distinct from the earlier FM-line hypotheses B/C (which just added auxiliary BCE
losses on the *same* shared embedding, no experts/gating at all). N shared expert
MLPs consume the same input (id embeddings + causal numeric features, DCN's x0
convention); each task (long_view + auxiliary is_click/is_like/is_follow) has its
own softmax gate over the experts plus its own small tower, so tasks can weight
the shared experts differently instead of forcing one representation to serve
every task equally.
"""
import torch
import torch.nn as nn


def best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class MMoE(nn.Module):
    def __init__(self, vocab_dim, num_fields, num_numeric, tasks, emb_dim=16,
                 n_experts=4, expert_dims=(64, 32), tower_dims=(16,), dropout=0.2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(vocab_dim, emb_dim, generator=g) * 0.01)
        self.num_fields = num_fields
        self.tasks = list(tasks)
        input_dim = num_fields * emb_dim + num_numeric

        def make_mlp(prev, dims, final_dim=None):
            layers = []
            for h in dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            if final_dim is not None:
                layers.append(nn.Linear(prev, final_dim))
            return nn.Sequential(*layers), prev

        self.experts = nn.ModuleList()
        expert_out = expert_dims[-1]
        for _ in range(n_experts):
            mlp, _ = make_mlp(input_dim, expert_dims)
            self.experts.append(mlp)

        self.gates = nn.ModuleDict({t: nn.Linear(input_dim, n_experts) for t in self.tasks})
        self.towers = nn.ModuleDict()
        for t in self.tasks:
            tower, _ = make_mlp(expert_out, tower_dims, final_dim=1)
            self.towers[t] = tower

    def forward(self, X, num_x):
        emb = self.V[X]                                    # (B, F, k)
        e_flat = emb.reshape(emb.shape[0], -1)
        x0 = torch.cat([e_flat, num_x], dim=1)              # (B, input_dim)

        expert_outs = torch.stack([e(x0) for e in self.experts], dim=1)  # (B, n_experts, expert_out)
        logits = {}
        for t in self.tasks:
            gate = torch.softmax(self.gates[t](x0), dim=1).unsqueeze(-1)  # (B, n_experts, 1)
            gated = (gate * expert_outs).sum(dim=1)                       # (B, expert_out)
            logits[t] = self.towers[t](gated).squeeze(-1)
        return logits

    def state_snapshot(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_snapshot(self, snap):
        self.load_state_dict(snap)
