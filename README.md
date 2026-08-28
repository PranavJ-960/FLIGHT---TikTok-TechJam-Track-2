# KuaiRand-Pure Autonomous ML Research Agent

TechJam Track 2: an agent that reproduces the official baseline, then
autonomously iterates to beat it, on KuaiRand-Pure within-user ranking.

## Task definition (fixed by the organizers — do not change)

| | |
|---|---|
| Task | Within-user ranking — rank each user's own impressions in the eval set; not full-catalog retrieval |
| Label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary = mean of both** |
| Split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Convergence rule | ε = 0.002, N = 3 (stop when primary improves ≤ ε for N consecutive iterations) |

See [`starter_kit/evaluate.py`](starter_kit/evaluate.py) — the scoring convention is fixed there, do not modify it.

## Baseline to beat

Official baseline is a from-scratch Factorization Machine (`starter_kit/baseline.py`, NumPy only).
Verified against `starter_kit/baseline_scores.json` on 2026-08-28 — real data, seed 0:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity check) | 0.4999 | 0.4514 | 0.4757 |
| item popularity | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** (published) / 0.6621 (ours) | **0.5282** / 0.5286 | **0.5946** / 0.5953 |

Reproduce:

```bash
python3 starter_kit/baseline.py --data_dir ./data/KuaiRand-Pure/data --model fm
```

## Setup

```bash
pip install numpy   # baseline itself needs nothing else
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzf KuaiRand-Pure.tar.gz -C ./data/
```

## Project layout

```
starter_kit/          # official reference — do not edit
  data.py              # data loading, official split, feature encoding
  evaluate.py           # GAUC / nDCG@5 scoring — the only source of truth for the metric
  baseline.py            # random / item-popularity / FM baselines
  submit.py                # generate + validate submission CSVs
  ablation_features.py       # reproduces the organizers' "extra features = no gain" result
  baseline_scores.json         # published scores, seed variance, convergence rule
data/KuaiRand-Pure/            # real dataset (gitignored, download per Setup above)
```

## Known dead ends (already tested by the organizers — don't repeat)

- Adding CWM's 13 static feature domains: no gain (0.5940 vs 0.5950).
- Bigger embedding dims (k = 8/16/32): no gain (~0.589 flat).
- Bottleneck is not features or model capacity — `user_id × video_id` already captures
  most learnable signal, and pure user-side features contribute nothing to a
  within-user ranking (they're constant within a user's group).

## Where headroom likely is (organizers' ranked guess, untested by them)

1. Pairwise/listwise loss instead of pointwise logloss — the metric is a ranking metric,
   the current loss isn't.
2. User history sequences (DIN/SIM-style) — completely unused currently.
3. Multi-task with `is_click`/`is_like`/`is_follow`/etc. as auxiliary signals for `long_view`.
4. Censored watch-time regression, à la [CWM](https://github.com/hyz20/CWM) — a research-depth
   direction, not a starting point (CWM targets a different label/loss and needs torch 1.6.0).
5. Other backbones (DeepFM/DCN/xDeepFM) — lower priority since capacity isn't the bottleneck.

## History

An earlier iteration of this repo (commit `ceae8ad`) built a from-scratch pipeline against
NDCG@10/Recall@50 on `click`, which is **not** the actual scored metric — replaced 2026-08-28
after cross-checking the official starter kit. See git history if any of that logic is useful
for reference (feature ideas, agent-loop structure), but rebuild it against `starter_kit/`'s
`long_view` / GAUC+nDCG@5 / date-based split before reusing.
