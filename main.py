"""
main.py
-------
End-to-end entry point:

    python main.py --data_dir ./data/KuaiRand-Pure/data

Steps:
1. Load train/val/test splits (data_loader.py) + merge side features.
2. Train the "official baseline" candidate to get baseline_metrics.
3. Run the autonomous agent loop (agent_loop.py) which iterates feature
   pipelines / hyperparameters, logging every attempt to logs/run_log.jsonl.
4. Score the best candidate ONCE on the held-out test split.
5. Print + save a results table (results_summary.json) with the delta vs.
   baseline, matching the challenge's Final Submission requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from data_loader import load_kuairand_pure, merge_features, LABEL_COL
from features import build_features
from model import train_model
from evaluate import evaluate_ranking, compute_delta_vs_baseline
from agent_loop import CANDIDATE_POOL, autonomous_agent_loop, run_iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/KuaiRand-Pure/data",
                         help="Path to the extracted KuaiRand-Pure/data folder")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--max_seconds", type=float, default=1800.0,
                         help="Wall-clock budget for the agent loop")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    run_log_path = os.path.join(args.log_dir, "run_log.jsonl")
    # start each run with a fresh log file
    open(run_log_path, "w").close()

    print(f"[1/5] Loading KuaiRand-Pure from {args.data_dir} ...")
    bundle = load_kuairand_pure(args.data_dir)

    train_raw = merge_features(bundle.train, bundle.user_features, bundle.video_basic, bundle.video_stat)
    val_raw = merge_features(bundle.val, bundle.user_features, bundle.video_basic, bundle.video_stat)
    test_raw = merge_features(bundle.test, bundle.user_features, bundle.video_basic, bundle.video_stat)

    print(f"    train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    print("[2/5] Training official-baseline-equivalent candidate ...")
    baseline_candidate = CANDIDATE_POOL[0]
    baseline_result = run_iteration(baseline_candidate, train_raw, val_raw, label_col=LABEL_COL)
    baseline_metrics = baseline_result["metrics"]
    print(f"    baseline val metrics: {baseline_metrics}")

    print(f"[3/5] Running autonomous agent loop (budget={args.max_seconds}s) ...")
    t0 = time.time()
    loop_result = autonomous_agent_loop(
        train_raw, val_raw,
        baseline_metrics=baseline_metrics,
        log_path=run_log_path,
        candidates=CANDIDATE_POOL,
        max_seconds=args.max_seconds,
    )
    elapsed = time.time() - t0
    print(f"    agent loop finished in {elapsed:.1f}s, "
          f"{len(loop_result['history'])} iterations attempted")

    best = loop_result["best_result"]
    if best is None:
        print("    WARNING: no candidate succeeded; falling back to baseline.")
        best_candidate = baseline_candidate
        best_feature_pipeline = baseline_candidate.feature_pipeline
        best_model = baseline_result["model"]
    else:
        best_candidate = best["candidate"]
        best_feature_pipeline = best_candidate.feature_pipeline
        best_model = best["model"]
        print(f"    best candidate: '{best_candidate.name}' "
              f"(val score_dataset delta = {best['delta_vs_baseline']['score_dataset']:.4f})")

    print("[4/5] Scoring the selected model ONCE on the held-out test split ...")
    test_feat, _ = build_features(test_raw, best_feature_pipeline)
    test_feat["pred"] = best_model.predict(test_feat)
    test_metrics = evaluate_ranking(test_feat, score_col="pred", label_col=LABEL_COL)

    # Baseline also needs a test-set score for a fair delta report.
    baseline_test_feat, _ = build_features(test_raw, baseline_candidate.feature_pipeline)
    baseline_test_feat["pred"] = baseline_result["model"].predict(baseline_test_feat)
    baseline_test_metrics = evaluate_ranking(baseline_test_feat, score_col="pred", label_col=LABEL_COL)

    test_deltas = compute_delta_vs_baseline(test_metrics, baseline_test_metrics)

    print("[5/5] Writing results_summary.json ...")
    summary = {
        "selected_candidate": best_candidate.name,
        "selected_hypothesis": best_candidate.hypothesis,
        "baseline_test_metrics": baseline_test_metrics,
        "agent_test_metrics": test_metrics,
        "test_delta_vs_baseline": test_deltas,
        "n_iterations_attempted": len(loop_result["history"]),
        "manual_interventions": loop_result["run_summary"]["manual_interventions"],
        "elapsed_seconds": elapsed,
    }
    out_path = os.path.join(args.output_dir, "results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nFull run log: {run_log_path}")
    print(f"Results summary: {out_path}")


if __name__ == "__main__":
    main()
