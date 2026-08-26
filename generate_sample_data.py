"""
generate_sample_data.py
------------------------
Creates a small synthetic dataset matching the KuaiRand-Pure schema exactly
(same file names, same columns) so you can smoke-test the full pipeline
(data_loader -> features -> model -> evaluate -> agent_loop) before
downloading the real 194MB dataset from Zenodo.

This is NOT real data and produces meaningless metrics -- it only exists to
verify the code runs end-to-end. Once you've confirmed that, download the
real dataset (see README.md) and point --data_dir at it.

Usage:
    python generate_sample_data.py --out_dir ./data/KuaiRand-Pure/data
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def make_logs(n_users: int, n_videos: int, n_rows: int, start_date: int, end_date: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(0, n_users, size=n_rows)
    video_ids = rng.integers(0, n_videos, size=n_rows)
    dates = rng.integers(start_date, end_date + 1, size=n_rows)
    hourmin = rng.integers(0, 2400, size=n_rows)
    time_ms = (dates.astype(np.int64) * 86_400_000) + rng.integers(0, 86_400_000, size=n_rows)

    # give videos an intrinsic "quality" so the model has real signal to learn
    video_quality = rng.beta(2, 8, size=n_videos)
    click_prob = video_quality[video_ids] * 0.6 + rng.uniform(0, 0.1, size=n_rows)
    is_click = rng.binomial(1, np.clip(click_prob, 0, 1))

    duration_ms = rng.integers(3000, 60000, size=n_rows)
    play_time_ms = np.where(
        is_click == 1,
        (duration_ms * rng.uniform(0.3, 1.2, size=n_rows)).astype(int),
        (duration_ms * rng.uniform(0, 0.3, size=n_rows)).astype(int),
    )
    long_view = (play_time_ms >= np.minimum(duration_ms, 18000)).astype(int)

    df = pd.DataFrame({
        "user_id": user_ids,
        "video_id": video_ids,
        "date": dates,
        "hourmin": hourmin,
        "time_ms": time_ms,
        "is_click": is_click,
        "is_like": rng.binomial(1, 0.05 * is_click, size=n_rows),
        "is_follow": rng.binomial(1, 0.01 * is_click, size=n_rows),
        "is_comment": rng.binomial(1, 0.01 * is_click, size=n_rows),
        "is_forward": rng.binomial(1, 0.005 * is_click, size=n_rows),
        "is_hate": rng.binomial(1, 0.01, size=n_rows),
        "long_view": long_view,
        "play_time_ms": play_time_ms,
        "duration_ms": duration_ms,
        "profile_stay_time": rng.integers(0, 5000, size=n_rows),
        "comment_stay_time": rng.integers(0, 3000, size=n_rows),
        "is_profile_enter": rng.binomial(1, 0.02, size=n_rows),
        "is_rand": np.zeros(n_rows, dtype=int),
        "tab": rng.integers(0, 15, size=n_rows),
    })
    return df


def make_user_features(n_users: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    activity_levels = ["high_active", "full_active", "middle_active", "UNKNOWN"]
    follow_ranges = ["0", "(0,10]", "(10,50]", "(50,100]", "(100,150]", "500+"]
    fans_ranges = ["0", "[1,10)", "[10,100)", "[100,1k)", "[1k,5k)"]
    friend_ranges = ["0", "[1,5)", "[5,30)", "[30,60)", "250+"]
    register_ranges = ["15-30", "31-60", "61-90", "91-180", "181-365", "366-730", "730+"]

    df = pd.DataFrame({"user_id": np.arange(n_users)})
    df["user_active_degree"] = rng.choice(activity_levels, size=n_users)
    df["is_lowactive_period"] = rng.binomial(1, 0.1, size=n_users)
    df["is_live_streamer"] = rng.binomial(1, 0.02, size=n_users)
    df["is_video_author"] = rng.binomial(1, 0.3, size=n_users)
    df["follow_user_num"] = rng.integers(0, 500, size=n_users)
    df["follow_user_num_range"] = rng.choice(follow_ranges, size=n_users)
    df["fans_user_num"] = rng.integers(0, 5000, size=n_users)
    df["fans_user_num_range"] = rng.choice(fans_ranges, size=n_users)
    df["friend_user_num"] = rng.integers(0, 250, size=n_users)
    df["friend_user_num_range"] = rng.choice(friend_ranges, size=n_users)
    df["register_days"] = rng.integers(15, 2000, size=n_users)
    df["register_days_range"] = rng.choice(register_ranges, size=n_users)
    for i in range(18):
        df[f"onehot_feat{i}"] = rng.integers(0, 5, size=n_users)
    return df


def make_video_basic(n_videos: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"video_id": np.arange(n_videos)})
    df["author_id"] = rng.integers(0, n_videos // 2 + 1, size=n_videos)
    df["video_type"] = rng.choice(["NORMAL", "AD"], size=n_videos, p=[0.95, 0.05])
    df["upload_dt"] = "2022-01-01"
    df["upload_type"] = rng.choice(["ShortImport", "LongImport", "Live"], size=n_videos)
    df["visible_status"] = 1
    df["video_duration"] = rng.integers(3000, 60000, size=n_videos)
    df["server_width"] = 720
    df["server_height"] = 1280
    df["music_id"] = rng.integers(0, 100000, size=n_videos)
    df["music_type"] = rng.integers(0, 10, size=n_videos)
    df["tag"] = rng.integers(0, 100, size=n_videos).astype(str)
    return df


def make_video_stat(n_videos: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"video_id": np.arange(n_videos)})
    df["counts"] = rng.integers(1, 100, size=n_videos)
    df["show_cnt"] = rng.gamma(2, 20, size=n_videos)
    df["show_user_num"] = df["show_cnt"] * rng.uniform(0.8, 1.0, size=n_videos)
    df["play_cnt"] = df["show_cnt"] * rng.uniform(0.05, 0.4, size=n_videos)
    df["play_user_num"] = df["play_cnt"] * rng.uniform(0.8, 1.0, size=n_videos)
    df["play_duration"] = df["play_cnt"] * rng.uniform(3000, 20000, size=n_videos)
    df["complete_play_cnt"] = df["play_cnt"] * rng.uniform(0.01, 0.3, size=n_videos)
    df["complete_play_user_num"] = df["complete_play_cnt"]
    df["valid_play_cnt"] = df["play_cnt"] * rng.uniform(0.1, 0.6, size=n_videos)
    df["valid_play_user_num"] = df["valid_play_cnt"]
    df["long_time_play_cnt"] = df["play_cnt"] * rng.uniform(0.05, 0.4, size=n_videos)
    df["long_time_play_user_num"] = df["long_time_play_cnt"]
    df["short_time_play_cnt"] = df["play_cnt"] * rng.uniform(0.1, 0.5, size=n_videos)
    df["short_time_play_user_num"] = df["short_time_play_cnt"]
    df["play_progress"] = rng.uniform(0, 1, size=n_videos)
    df["comment_stay_duration"] = rng.gamma(1, 500, size=n_videos)
    df["like_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.1, size=n_videos)
    df["like_user_num"] = df["like_cnt"]
    df["comment_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.02, size=n_videos)
    df["comment_user_num"] = df["comment_cnt"]
    df["follow_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.01, size=n_videos)
    df["follow_user_num"] = df["follow_cnt"]
    df["share_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.01, size=n_videos)
    df["share_user_num"] = df["share_cnt"]
    df["download_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.02, size=n_videos)
    df["download_user_num"] = df["download_cnt"]
    df["report_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.001, size=n_videos)
    df["report_user_num"] = df["report_cnt"]
    df["collect_cnt"] = df["play_cnt"] * rng.uniform(0.0, 0.03, size=n_videos)
    df["collect_user_num"] = df["collect_cnt"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./data/KuaiRand-Pure/data")
    parser.add_argument("--n_users", type=int, default=2000)
    parser.add_argument("--n_videos", type=int, default=1500)
    parser.add_argument("--n_train_rows", type=int, default=60000)
    parser.add_argument("--n_valtest_rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating synthetic logs ...")
    train_log = make_logs(args.n_users, args.n_videos, args.n_train_rows,
                           start_date=20220408, end_date=20220421, seed=args.seed)
    valtest_log = make_logs(args.n_users, args.n_videos, args.n_valtest_rows,
                             start_date=20220422, end_date=20220508, seed=args.seed + 1)
    random_log = make_logs(args.n_users, args.n_videos, args.n_valtest_rows // 10,
                            start_date=20220422, end_date=20220508, seed=args.seed + 2)
    random_log["is_rand"] = 1

    print("Generating synthetic feature tables ...")
    user_features = make_user_features(args.n_users, args.seed)
    video_basic = make_video_basic(args.n_videos, args.seed)
    video_stat = make_video_stat(args.n_videos, args.seed)

    train_log.to_csv(os.path.join(args.out_dir, "log_standard_4_08_to_4_21_pure.csv"), index=False)
    valtest_log.to_csv(os.path.join(args.out_dir, "log_standard_4_22_to_5_08_pure.csv"), index=False)
    random_log.to_csv(os.path.join(args.out_dir, "log_random_4_22_to_5_08_pure.csv"), index=False)
    user_features.to_csv(os.path.join(args.out_dir, "user_features_pure.csv"), index=False)
    video_basic.to_csv(os.path.join(args.out_dir, "video_features_basic_pure.csv"), index=False)
    video_stat.to_csv(os.path.join(args.out_dir, "video_features_statistic_pure.csv"), index=False)

    print(f"Done. Synthetic KuaiRand-Pure-shaped data written to {args.out_dir}")


if __name__ == "__main__":
    main()
