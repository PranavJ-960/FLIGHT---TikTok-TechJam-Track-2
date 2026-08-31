"""BST (Behavior Sequence Transformer)-style model: a genuine self-attention
Transformer over the user's actual chronological history (experiments/data_seq.py,
max_len>1 — unlike DCN/DeepFM's DIN-style single-step attention, or iteration D's
earlier hypothesis-D attention). The target video is appended as the final sequence
token (BST's own trick: puts "should the model attend the candidate to history" and
"should it attend history to itself" in the same self-attention stack, instead of a
separate cross-attention module), with learned positional embeddings over the
(possibly padded) history positions. The final token's output (target-conditioned
on its attended history) is concatenated with the proven causal numeric features and
fed to an MLP head.
"""
import torch
import torch.nn as nn


def best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class BST(nn.Module):
    def __init__(self, vocab_dim, max_len, num_numeric, emb_dim=16, n_heads=2,
                 n_layers=2, mlp_dims=(64, 32), dropout=0.2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(vocab_dim + 1, emb_dim, generator=g) * 0.01)  # +1: PAD row
        self.pos = nn.Parameter(torch.randn(max_len + 1, emb_dim, generator=g) * 0.01)  # +1: target slot
        self.max_len = max_len

        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
                                            dropout=dropout, batch_first=True)
        # enable_nested_tensor=False: the nested-tensor fast path for padding masks
        # isn't implemented on MPS (aten::_nested_tensor_from_mask_left_aligned).
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

        mlp_layers = []
        prev = emb_dim + num_numeric
        for h in mlp_dims:
            mlp_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, target_vid, H, M, num_x):
        """target_vid: (B,) target video's slot in the shared vocab. H: (B, L) history
        video slots (PAD row for padding). M: (B, L) 1=real, 0=pad. num_x: (B, D) causal
        numeric features."""
        B = target_vid.shape[0]
        hist_emb = self.V[H] + self.pos[:self.max_len]                     # (B, L, k)
        tgt_emb = (self.V[target_vid] + self.pos[self.max_len]).unsqueeze(1)  # (B, 1, k)
        seq = torch.cat([hist_emb, tgt_emb], dim=1)                        # (B, L+1, k)

        pad_mask = torch.cat([M == 0, torch.zeros(B, 1, dtype=torch.bool, device=M.device)], dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad_mask)             # (B, L+1, k)
        tgt_out = out[:, -1, :]                                            # (B, k)

        combined = torch.cat([tgt_out, num_x], dim=1)
        return self.mlp(combined).squeeze(-1)

    def state_snapshot(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_snapshot(self, snap):
        self.load_state_dict(snap)
