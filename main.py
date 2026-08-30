"""
main.py
-------
End-to-end entry point:

    python main.py --data_dir ./data/KuaiRand-Pure/data

Steps:
1. Load the OFFICIAL train/valid/test splits (date-based) and the OFFICIAL
   label (long_view), then merge side features.
2. Run the autonomous agent loop on train + validation only, logging every
   iteration to logs/run_log.jsonl.
3. Score the validation-best candidate ONCE on the held-out test split.
4. Write submission.csv (row_id,user_id,video_id,score) and
   outputs/results_summary.json with the delta vs. the OFFICIAL published
   FM baseline and the resource usage the judging criteria ask for.

The test split is touched exactly once, in step 3. Everything before that
sees train + validation only.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from data_loader import load_kuairand_pure, merge_features, LABEL_COL
from candidate import build_candidate_frames
from agent_loop import autonomous_agent_loop, CANDIDATE_POOL, score_frame
from proposer import make_proposer
import scoring


def write_submission(df: pd.DataFrame, preds, path: str) -> str:
    """
    Official submission schema: row_id,user_id,video_id,score -- one line per
    evaluation-split row, row_id 0-based and strictly increasing in the order
    official data.load() produces. Validate afterwards with:
        cd official && python submit.py --check --split test ../submission.csv
    """
    out = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "user_id": df["user_id"].to_numpy(),
        "video_id": df["video_id"].to_numpy(),
        "score": preds,
    })
    out.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/KuaiRand-Pure/data")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--submission", default="./submission.csv")
    parser.add_argument("--max_seconds", type=float, default=scoring.MAX_SECONDS)
    parser.add_argument("--max_iterations", type=int, default=scoring.MAX_ITERATIONS)
    parser.add_argument("--static", action="store_true",
                        help="force the offline static proposer (no LLM calls)")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    run_log_path = os.path.join(args.log_dir, "run_log.jsonl")
    open(run_log_path, "w").close()   # fresh log per run

    print(f"[1/5] Loading KuaiRand-Pure (official split, label={LABEL_COL}) "
          f"from {args.data_dir} ...")
    bundle = load_kuairand_pure(args.data_dir)
    train_raw = merge_features(bundle.train, bundle.user_features,
                               bundle.video_basic, bundle.video_stat)
    val_raw = merge_features(bundle.val, bundle.user_features,
                             bundle.video_basic, bundle.video_stat)
    test_raw = merge_features(bundle.test, bundle.user_features,
                              bundle.video_basic, bundle.video_stat)
    print(f"    train={len(train_raw):,}  valid={len(val_raw):,}  test={len(test_raw):,}")

    print("[2/5] Official baseline to beat (organizer-published, not retrained):")
    print(f"    valid primary {scoring.FM_BASELINE_VALID['primary']:.4f} | "
          f"test primary {scoring.FM_BASELINE_TEST['primary']:.4f}")

    proposer = make_proposer(seed_pool=CANDIDATE_POOL, force_static=args.static)
    print(f"[3/5] Running autonomous agent loop "
          f"(proposer={proposer.source}, max {args.max_iterations} iters, "
          f"{args.max_seconds/3600:.1f}h budget, "
          f"eps={scoring.EPSILON}, N={scoring.PATIENCE_N}) ...")
    t0 = time.time()
    loop_result = autonomous_agent_loop(
        train_raw, val_raw,
        log_path=run_log_path,
        proposer=proposer,
        max_iterations=args.max_iterations,
        max_seconds=args.max_seconds,
    )
    elapsed = time.time() - t0
    n_iters = len(loop_result["history"])
    n_failed = sum(1 for r in loop_result["history"] if r.get("status") == "error")
    print(f"    finished in {elapsed:.1f}s over {n_iters} iterations "
          f"({n_failed} failed and recovered)")

    best = loop_result["best_result"]
    if best is None:
        raise SystemExit("No candidate completed successfully -- nothing to submit. "
                         "Check logs/run_log.jsonl for the recorded errors.")
    best_candidate = best["candidate"]
    print(f"    validation-best: '{best_candidate.name}' "
          f"(valid primary {best['metrics']['primary']:.4f})")

    print("[4/5] Scoring the selected model ONCE on the held-out test split ...")
    # Rebuild features for test through the winning candidate: fit() re-runs on
    # train (deterministic, same state) and apply() runs on test.
    _, (test_feat,), _ = build_candidate_frames(best_candidate, train_raw, test_raw)
    test_preds = best["model"].predict(test_feat)
    test_metrics = score_frame(test_feat, test_preds)
    test_deltas = scoring.delta_vs_official(test_metrics, split="test")
    print(f"    {scoring.summarize(test_metrics, 'test')}")

    sub_path = write_submission(test_feat, test_preds, args.submission)
    print(f"    submission written: {sub_path}")

    print("[5/5] Writing results_summary.json ...")
    summary = {
        "selected_candidate": best_candidate.name,
        "selected_hypothesis": best_candidate.hypothesis,
        "selected_code": best_candidate.code,
        "selected_params": best_candidate.param_overrides,
        "official_baseline_valid": scoring.FM_BASELINE_VALID,
        "official_baseline_test": scoring.FM_BASELINE_TEST,
        "agent_valid_metrics": best["metrics"],
        "agent_test_metrics": test_metrics,
        "test_delta_vs_official_baseline": test_deltas,
        "beats_official_baseline": bool(test_deltas["primary_delta"] > 0),
        "progress_vs_oracle_ceiling": scoring.progress_vs_oracle(test_metrics, "test"),
        "resource_usage": {
            "iterations_used": n_iters,
            "iterations_failed_and_recovered": n_failed,
            "iteration_cap": args.max_iterations,
            "agent_wall_clock_seconds": elapsed,
            **loop_result["run_summary"],
            "gpu_hours": 0.0,   # CPU-only pipeline
        },
    }
    out_path = os.path.join(args.output_dir, "results_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nRun log:  {run_log_path}")
    print(f"Results:  {out_path}")
    print(f"Validate: cd official && PYTHONIOENCODING=utf-8 python submit.py "
          f"--check --split test ../{os.path.basename(sub_path)} "
          f"--data_dir ../data/KuaiRand-Pure/data")


if __name__ == "__main__":
    main()
