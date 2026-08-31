"""Autonomous orchestrator for the causal-feature ranking pipeline (LightGBM / XGBoost
/ CatBoost, all with per-user weighting — see [[project-best-model-lgb]]).

No human chooses individual hypotheses here: each iteration is selected by the
orchestrator's own adaptive logic (cost- and success-weighted proposer sampling),
evaluated on valid only, and the loop stops itself using the competition's own rules:

  - converged: validation primary improves by <= epsilon over the last `patience`
    consecutive iterations
  - iteration cap (default 50)
  - wall-clock budget (configurable; the competition ceiling is 6h, but a proposer is
    only ever started if its estimated cost fits in the remaining budget, so the loop
    is wall-clock-aware by construction, not just cut off mid-run)

Every iteration — success, non-improvement, or error — is appended to
logs/orchestrator_run.json as it happens (crash loses at most the in-flight
iteration), and a human-readable logs/orchestrator_report.md is written at the end.
Robustness: each proposer call is wrapped in try/except; a failure is logged as a
recovery event and the loop continues with the next proposer, it never crashes the
run. Train/valid only — test is never unpacked or evaluated here.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoost, Pool

from starter_kit.evaluate import evaluate
from experiments.data_causal import encode_causal, CAT_FIELDS, NUMERIC_FIELDS
from experiments.tune_lgb import sample_params as sample_lgb_params
from experiments.tune_xgb import sample_params as sample_xgb_params
from experiments.tune_catboost import sample_params as sample_catboost_params

BASELINE_VALID_PRIMARY = 0.6015   # reproduced official FM baseline, mean seeds 0-2, valid
EPSILON = 0.002
PATIENCE = 3
MAX_ITERATIONS = 50

FEATURE_VARIANTS = [(s, w) for s in (1.0, 2.0, 5.0, 10.0, 20.0) for w in (10, 20, 40)]


# ---------------------------------------------------------------------------
# feature/dataset construction (rebuilt whenever a feature-variant iteration wins)
# ---------------------------------------------------------------------------

def build_lgb_matrix(enc, split):
    d = enc[split]
    X = np.concatenate([d['num'], d['dow'][:, None].astype(np.float32), d['cat'].astype(np.float32)], axis=1)
    return X, d['y'], d['users']


def build_cb_frame(enc, split):
    d = enc[split]
    df = pd.DataFrame(d['num'], columns=NUMERIC_FIELDS)
    df['dow'] = d['dow'].astype(str)
    for i, c in enumerate(CAT_FIELDS):
        df[c] = d['cat'][:, i].astype(str)
    return df, d['y'], d['users']


def build_active_dataset(data_dir, smooth, recent_window):
    enc, feature_names = encode_causal(data_dir, smooth=smooth, recent_window=recent_window)

    Xtr, ytr, utr = build_lgb_matrix(enc, 'train')
    Xva, yva, uva = build_lgb_matrix(enc, 'valid')
    cat_idx = [feature_names.index(c) for c in (['dow'] + CAT_FIELDS)]
    order_tr = np.argsort(utr, kind='stable')
    order_va = np.argsort(uva, kind='stable')
    utr_s = np.array([utr[i] for i in order_tr])
    uva_s = np.array([uva[i] for i in order_va])
    _, group_tr = np.unique(utr_s, return_counts=True)
    _, group_va = np.unique(uva_s, return_counts=True)
    group_map = dict(zip(*np.unique(utr_s, return_counts=True)))
    row_w_tr = np.array([1.0 / group_map[u] for u in utr_s], dtype=np.float32)
    group_w_tr = (1.0 / group_tr).astype(np.float32)

    Xtr_s, ytr_s = Xtr[order_tr], ytr[order_tr]
    Xva_s, yva_s = Xva[order_va], yva[order_va]
    lgb_train = lgb.Dataset(Xtr_s, label=ytr_s, weight=row_w_tr, group=group_tr,
                             feature_name=feature_names, categorical_feature=cat_idx)
    lgb_valid = lgb.Dataset(Xva_s, label=yva_s, group=group_va, feature_name=feature_names,
                             categorical_feature=cat_idx, reference=lgb_train)

    xgb_train = xgb.DMatrix(Xtr_s, label=ytr_s, feature_names=feature_names)
    xgb_train.set_weight(group_w_tr)
    xgb_train.set_group(group_tr)
    # separate, UNWEIGHTED eval set for early stopping — reusing the (weighted) train set
    # as its own eval target crashes XGBoost's NDCG metric ("Invalid output score, might be
    # caused by invalid query group weight"), caught live by the orchestrator's own error
    # handling on the first real run and fixed here.
    xgb_valid = xgb.DMatrix(Xva_s, label=yva_s, feature_names=feature_names)
    xgb_valid.set_group(group_va)
    xgb_valid_full = xgb.DMatrix(Xva, feature_names=feature_names)

    Ctr, cytr, cutr = build_cb_frame(enc, 'train')
    Cva, cyva, cuva = build_cb_frame(enc, 'valid')
    cat_cols = ['dow'] + CAT_FIELDS
    order_ctr = np.argsort(cutr, kind='stable')
    order_cva = np.argsort(cuva, kind='stable')
    group_ctr = np.array([cutr[i] for i in order_ctr])
    group_cva = np.array([cuva[i] for i in order_cva])
    # CatBoost's group_weight is per-ROW (constant within a group), unlike XGBoost's
    # per-GROUP `group_w_tr` above — a real bug caught by the orchestrator's own smoke test.
    row_w_ctr = np.array([1.0 / group_map[u] for u in group_ctr], dtype=np.float32)
    cb_train = Pool(Ctr.iloc[order_ctr], label=cytr[order_ctr], group_id=group_ctr,
                     group_weight=row_w_ctr, cat_features=cat_cols)
    cb_valid = Pool(Cva.iloc[order_cva], label=cyva[order_cva], group_id=group_cva, cat_features=cat_cols)
    cb_valid_full = Pool(Cva, cat_features=cat_cols)

    return {
        'smooth': smooth, 'recent_window': recent_window, 'feature_names': feature_names,
        'lgb_train': lgb_train, 'lgb_valid': lgb_valid, 'Xva_lgb': Xva,
        'xgb_train': xgb_train, 'xgb_valid': xgb_valid, 'xgb_valid_full': xgb_valid_full,
        'cb_train': cb_train, 'cb_valid': cb_valid, 'cb_valid_full': cb_valid_full,
        'uva': uva, 'yva': yva,
    }


# ---------------------------------------------------------------------------
# proposers: each returns (hypothesis:str, est_cost_s:float, run_fn:callable)
# run_fn() returns {'primary':, 'GAUC':, 'nDCG@5':, 'scores': np.array, 'family': str, 'params': dict}
# ---------------------------------------------------------------------------

def propose_lgb(state, rng):
    ds = state['dataset']
    params = sample_lgb_params(rng)
    seed = int(rng.integers(0, 10_000))

    def run():
        run_params = dict(params, objective='lambdarank', metric='ndcg', eval_at=[5],
                           verbosity=-1, seed=seed, feature_pre_filter=False)
        bst = lgb.train(run_params, ds['lgb_train'], num_boost_round=500, valid_sets=[ds['lgb_valid']],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        scores = bst.predict(ds['Xva_lgb'], num_iteration=bst.best_iteration)
        m = evaluate(ds['uva'], ds['yva'], scores)
        return {**m, 'scores': scores, 'family': 'lgb', 'params': params, 'seed': seed}

    est_cost = state['cost_est'].get('lgb', 10.0)
    return f"LightGBM trial: {params}", est_cost, run


def propose_xgb(state, rng):
    ds = state['dataset']
    params = sample_xgb_params(rng)
    seed = int(rng.integers(0, 10_000))

    def run():
        run_params = dict(params, objective='rank:ndcg', eval_metric='ndcg@5', seed=seed, verbosity=0)
        bst = xgb.train(run_params, ds['xgb_train'], num_boost_round=1000,
                         evals=[(ds['xgb_valid'], 'valid')], early_stopping_rounds=30, verbose_eval=False)
        scores = bst.predict(ds['xgb_valid_full'], iteration_range=(0, bst.best_iteration + 1))
        m = evaluate(ds['uva'], ds['yva'], scores)
        return {**m, 'scores': scores, 'family': 'xgb', 'params': params, 'seed': seed}

    est_cost = state['cost_est'].get('xgb', 15.0)
    return f"XGBoost trial: {params}", est_cost, run


def propose_catboost(state, rng):
    ds = state['dataset']
    params = sample_catboost_params(rng)
    seed = int(rng.integers(0, 10_000))

    def run():
        run_params = dict(params, loss_function='YetiRank', eval_metric='NDCG:top=5',
                           iterations=400, random_seed=seed, verbose=False,
                           early_stopping_rounds=20, thread_count=-1)
        model = CatBoost(run_params)
        model.fit(ds['cb_train'], eval_set=ds['cb_valid'], use_best_model=True)
        scores = model.predict(ds['cb_valid_full'])
        m = evaluate(ds['uva'], ds['yva'], scores)
        return {**m, 'scores': scores, 'family': 'catboost', 'params': params, 'seed': seed}

    est_cost = state['cost_est'].get('catboost', 150.0)
    return f"CatBoost trial: {params}", est_cost, run


def propose_feature_variant(state, rng):
    untried = [fv for fv in FEATURE_VARIANTS if fv not in state['tried_feature_variants']]
    if not untried:
        return None
    smooth, window = untried[int(rng.integers(0, len(untried)))]
    data_dir = state['data_dir']
    default_lgb_params = {'learning_rate': 0.08, 'num_leaves': 63, 'min_data_in_leaf': 30,
                           'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'lambda_l1': 0.1,
                           'lambda_l2': 1.0, 'max_depth': 6, 'min_gain_to_split': 0.01}

    def run():
        new_ds = build_active_dataset(data_dir, smooth, window)
        run_params = dict(default_lgb_params, objective='lambdarank', metric='ndcg', eval_at=[5],
                           verbosity=-1, seed=0, feature_pre_filter=False)
        bst = lgb.train(run_params, new_ds['lgb_train'], num_boost_round=500, valid_sets=[new_ds['lgb_valid']],
                         callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        scores = bst.predict(new_ds['Xva_lgb'], num_iteration=bst.best_iteration)
        m = evaluate(new_ds['uva'], new_ds['yva'], scores)
        return {**m, 'scores': scores, 'family': 'lgb', 'params': default_lgb_params,
                'feature_variant': (smooth, window), 'new_dataset': new_ds}

    return f"feature variant smooth={smooth} recent_window={window}", 20.0, run


def propose_blend(state, rng):
    fam = state['family_best']
    available = [k for k in ('lgb', 'xgb', 'catboost') if fam.get(k) is not None]
    if len(available) < 2:
        return None

    def run():
        norm = {k: (fam[k]['scores'] - fam[k]['scores'].min()) /
                   (fam[k]['scores'].max() - fam[k]['scores'].min() + 1e-12) for k in available}
        ds = state['dataset']
        best_local = None
        steps = [i / 10 for i in range(11)]
        # 2-way if only 2 families, coarse 3-way simplex if all 3 are available
        if len(available) == 2:
            a, b = available
            for w in steps:
                blended = w * norm[a] + (1 - w) * norm[b]
                m = evaluate(ds['uva'], ds['yva'], blended)
                if best_local is None or m['primary'] > best_local[0]:
                    best_local = (m['primary'], m, blended, {a: w, b: 1 - w})
        else:
            a, b, c = available
            for wa in steps:
                for wb in steps:
                    wc = round(1.0 - wa - wb, 2)
                    if wc < -1e-9 or wc > 1 + 1e-9:
                        continue
                    wc = max(0.0, wc)
                    blended = wa * norm[a] + wb * norm[b] + wc * norm[c]
                    m = evaluate(ds['uva'], ds['yva'], blended)
                    if best_local is None or m['primary'] > best_local[0]:
                        best_local = (m['primary'], m, blended, {a: wa, b: wb, c: wc})
        _, m, scores, weights = best_local
        return {**m, 'scores': scores, 'family': 'blend', 'params': weights}

    return f"blend of {available}", 2.0, run


PROPOSERS = {'lgb': propose_lgb, 'xgb': propose_xgb, 'catboost': propose_catboost,
             'feature_variant': propose_feature_variant, 'blend': propose_blend}


def select_proposer(state, remaining_budget, rng):
    candidates = []
    for name, weight in state['weights'].items():
        est = state['cost_est'].get(name, {'lgb': 10, 'xgb': 15, 'catboost': 150,
                                            'feature_variant': 20, 'blend': 2}[name])
        if est * 1.15 <= remaining_budget:   # small safety margin
            candidates.append((name, weight, est))
    if not candidates:
        return None

    # guarantee baseline exploration breadth before allowing adaptive/weighted selection —
    # otherwise an early unlucky run of repeated samples from one (especially slow, high-
    # variance) proposer can trip the convergence patience counter before the loop has even
    # tried the other model families or a blend once. Cheapest untried proposer first.
    untried = [c for c in candidates if c[0] not in state['tried_types']]
    if untried:
        untried.sort(key=lambda c: c[2])
        return untried[0][0]

    names = [c[0] for c in candidates]
    weights = np.array([c[1] for c in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    return names[rng.choice(len(names), p=weights)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/KuaiRand-Pure/data')
    ap.add_argument('--wall_clock_budget_s', type=float, default=1800)
    ap.add_argument('--max_iterations', type=int, default=MAX_ITERATIONS)
    ap.add_argument('--epsilon', type=float, default=EPSILON)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='outputs')
    a = ap.parse_args()
    os.makedirs('logs', exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print(f"=== autonomous orchestrator: budget={a.wall_clock_budget_s}s, "
          f"max_iter={a.max_iterations}, epsilon={a.epsilon}, patience={a.patience} ===")
    t_start = time.time()
    print("building initial dataset (smooth=5.0, recent_window=20 — current best-known) ...")
    dataset = build_active_dataset(a.data_dir, smooth=5.0, recent_window=20)
    print(f"  ready at {time.time()-t_start:.1f}s")

    state = {
        'data_dir': a.data_dir,
        'dataset': dataset,
        'weights': {k: 1.0 for k in PROPOSERS},
        'cost_est': {},
        'family_best': {'lgb': None, 'xgb': None, 'catboost': None},
        'tried_feature_variants': {(dataset['smooth'], dataset['recent_window'])},
        'tried_types': set(),
    }

    best_primary = BASELINE_VALID_PRIMARY
    best_entry = None
    stall = 0
    history = []
    stop_reason = None
    it = 0

    while it < a.max_iterations:
        elapsed = time.time() - t_start
        remaining = a.wall_clock_budget_s - elapsed
        if remaining <= 5:
            stop_reason = 'wall_clock_budget_exceeded'
            break

        proposer_name = select_proposer(state, remaining, rng)
        if proposer_name is None:
            stop_reason = 'wall_clock_budget_exceeded (no proposer fits remaining time)'
            break

        proposal = PROPOSERS[proposer_name](state, rng)
        if proposal is None:   # e.g. feature_variant exhausted, or blend not yet available
            if proposer_name != 'blend':   # 'not enough families yet' isn't a signal blend
                state['weights'][proposer_name] *= 0.3   # is unpromising — don't penalize it for that
            # mark tried even on a no-op result, otherwise a cheap-but-not-yet-available
            # proposer (blend, before 2 families exist) would keep winning the "cheapest
            # untried" tiebreak forever and starve the others of their guaranteed first try
            state['tried_types'].add(proposer_name)
            continue

        hypothesis, est_cost, run_fn = proposal
        state['tried_types'].add(proposer_name)
        t0 = time.time()
        entry = {'iteration': it, 'proposer': proposer_name, 'hypothesis': hypothesis}
        try:
            result = run_fn()
            dt = time.time() - t0
            state['cost_est'][proposer_name] = (state['cost_est'].get(proposer_name, dt) + dt) / 2
            entry.update({'status': 'ok', 'seconds': round(dt, 1),
                           'valid': {'GAUC': result['GAUC'], 'nDCG@5': result['nDCG@5'], 'primary': result['primary']}})

            fam = result['family']
            if fam in state['family_best'] and (state['family_best'][fam] is None or
                                                 result['primary'] > state['family_best'][fam]['primary']):
                state['family_best'][fam] = {'primary': result['primary'], 'scores': result['scores'],
                                              'params': result.get('params')}

            if 'feature_variant' in result:
                state['tried_feature_variants'].add(result['feature_variant'])
                if result['primary'] > best_primary:
                    print(f"  [{it}] feature variant {result['feature_variant']} improved on active dataset "
                          f"({result['primary']:.4f} > {best_primary:.4f}) — switching active features")
                    state['dataset'] = result['new_dataset']
                    state['family_best'] = {'lgb': {'primary': result['primary'], 'scores': result['scores'],
                                                     'params': result['params']}, 'xgb': None, 'catboost': None}

            improved = result['primary'] > best_primary + a.epsilon
            if result['primary'] > best_primary:
                best_primary = result['primary']
                best_entry = {**entry, 'family': fam, 'params': result.get('params')}
                np.save(os.path.join(a.out_dir, 'orchestrator_best_valid_scores.npy'), result['scores'])
            stall = 0 if improved else stall + 1
            state['weights'][proposer_name] *= 1.3 if improved else 0.95

            print(f"  [{it:2d}] {proposer_name:16s} primary={result['primary']:.4f} "
                  f"(best={best_primary:.4f}, stall={stall}/{a.patience}) {dt:.1f}s")
        except Exception as e:
            dt = time.time() - t0
            entry.update({'status': 'error', 'seconds': round(dt, 1),
                           'error': str(e), 'traceback': traceback.format_exc(limit=5)})
            stall += 1
            state['weights'][proposer_name] *= 0.5
            print(f"  [{it:2d}] {proposer_name:16s} FAILED ({dt:.1f}s): {e} — logged, continuing")

        history.append(entry)
        with open('logs/orchestrator_run.json', 'w') as fh:
            json.dump({'config': vars(a), 'baseline_valid_primary': BASELINE_VALID_PRIMARY,
                        'history': history, 'best_so_far': best_primary}, fh, indent=2, default=float)

        it += 1
        if stall >= a.patience:
            stop_reason = 'converged'
            break

    if stop_reason is None:
        stop_reason = 'iteration_cap_reached'

    total_time = time.time() - t_start
    print(f"\n=== stopped: {stop_reason} | iterations={it} | wall_clock={total_time:.1f}s | "
          f"best valid primary={best_primary:.4f} (delta vs baseline {best_primary-BASELINE_VALID_PRIMARY:+.4f}) ===")

    with open('logs/orchestrator_run.json', 'w') as fh:
        json.dump({'config': vars(a), 'baseline_valid_primary': BASELINE_VALID_PRIMARY,
                    'history': history, 'stop_reason': stop_reason, 'iterations_run': it,
                    'wall_clock_s': round(total_time, 1), 'best_primary': best_primary,
                    'best_entry': best_entry}, fh, indent=2, default=float)

    with open('logs/orchestrator_report.md', 'w') as fh:
        fh.write(f"# Autonomous orchestrator run report\n\n")
        fh.write(f"- Stop reason: **{stop_reason}**\n")
        fh.write(f"- Iterations run: {it} / {a.max_iterations} cap\n")
        fh.write(f"- Wall-clock: {total_time:.1f}s (budget {a.wall_clock_budget_s}s)\n")
        fh.write(f"- Baseline valid primary: {BASELINE_VALID_PRIMARY:.4f}\n")
        fh.write(f"- Best valid primary found: {best_primary:.4f} "
                 f"(delta {best_primary-BASELINE_VALID_PRIMARY:+.4f})\n")
        if best_entry:
            fh.write(f"- Best iteration: #{best_entry['iteration']} via {best_entry['proposer']} "
                     f"({best_entry.get('family')})\n- Params: `{best_entry.get('params')}`\n")
        fh.write(f"- Errors encountered and recovered from: {sum(1 for e in history if e['status']=='error')}\n\n")
        fh.write("## Iteration log\n\n")
        for e in history:
            if e['status'] == 'ok':
                fh.write(f"- **[{e['iteration']}]** {e['hypothesis']} -> "
                         f"primary={e['valid']['primary']:.4f} ({e['seconds']}s)\n")
            else:
                fh.write(f"- **[{e['iteration']}]** {e['hypothesis']} -> "
                         f"FAILED: {e['error']} ({e['seconds']}s) — logged, run continued\n")
    print("logged to logs/orchestrator_run.json and logs/orchestrator_report.md")


if __name__ == '__main__':
    main()
