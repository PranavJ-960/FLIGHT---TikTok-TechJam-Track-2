"""
data_loader.py
--------------
Loads KuaiRand-Pure using the OFFICIAL split and the OFFICIAL label, but
returns pandas DataFrames so the LightGBM feature pipeline can keep working.

Why this file exists at all: official/data.py is numpy-only and returns
tuples with just 5 encoded fields. We want the full side-feature tables for
feature engineering, so we re-load with pandas -- but we take the split
boundaries and the label name FROM official/data.py so there is exactly one
source of truth for them.

Pinned conventions (from official/data.py and the Starter Kit README):
    label  : long_view   (NOT is_click)
    train  : date 20220408-20220421
    valid  : date 20220422-20220428
    test   : date 20220429-20220508     <- touch once, at the very end

Row order is reproduced exactly as official `data.load()` produces it --
read log_standard_4_08_to_4_21_pure.csv first, then
log_standard_4_22_to_5_08_pure.csv, filter by date, preserve file order.
That ordering defines `row_id` in the submission file, so it must match.
Use `verify_alignment()` to prove it does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from official.data import LABEL as LABEL_COL, SPLITS

# Read in this order -- it defines row_id.
LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
LOG_RANDOM_FILE = "log_random_4_22_to_5_08_pure.csv"  # randomised exposure; unbiased validation

USER_FEATURES_FILE = "user_features_pure.csv"
VIDEO_BASIC_FILE = "video_features_basic_pure.csv"
VIDEO_STAT_FILE = "video_features_statistic_pure.csv"


@dataclass
class KuaiRandData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    user_features: pd.DataFrame | None
    video_basic: pd.DataFrame | None
    video_stat: pd.DataFrame | None


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            "Download KuaiRand-Pure from "
            "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz "
            "and extract it so that its `data/` folder matches --data_dir."
        )
    return pd.read_csv(path)


def load_all_logs(data_dir: str) -> pd.DataFrame:
    """Concatenate both standard logs in the official order, adding row_id
    per split later. Order here is load order, not sorted by date."""
    frames = [_read_csv_safe(os.path.join(data_dir, f)) for f in LOG_FILES]
    return pd.concat(frames, ignore_index=True)


def split_by_date(all_logs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply the official date boundaries. Preserves load order within a
    split, then assigns row_id 0..n-1 -- this is the submission's row_id."""
    out = {}
    for name, (lo, hi) in SPLITS.items():
        mask = (all_logs["date"] >= lo) & (all_logs["date"] <= hi)
        part = all_logs[mask].reset_index(drop=True)
        part.insert(0, "row_id", range(len(part)))
        out[name] = part
    return out


def load_feature_tables(data_dir: str):
    return (
        _read_csv_safe(os.path.join(data_dir, USER_FEATURES_FILE)),
        _read_csv_safe(os.path.join(data_dir, VIDEO_BASIC_FILE)),
        _read_csv_safe(os.path.join(data_dir, VIDEO_STAT_FILE)),
    )


def load_kuairand_pure(data_dir: str, load_features: bool = True) -> KuaiRandData:
    """Main entry point: official splits, official label, pandas frames."""
    splits = split_by_date(load_all_logs(data_dir))

    user_features = video_basic = video_stat = None
    if load_features:
        user_features, video_basic, video_stat = load_feature_tables(data_dir)

    return KuaiRandData(
        train=splits["train"], val=splits["valid"], test=splits["test"],
        user_features=user_features, video_basic=video_basic, video_stat=video_stat,
    )


def merge_features(log_df, user_features, video_basic, video_stat) -> pd.DataFrame:
    """Left-join user/video side information onto an interaction log.
    Row order and row_id are preserved."""
    df = log_df.copy()
    if user_features is not None:
        df = df.merge(user_features, on="user_id", how="left", suffixes=("", "_user"))
    if video_basic is not None:
        df = df.merge(video_basic, on="video_id", how="left", suffixes=("", "_vbasic"))
    if video_stat is not None:
        df = df.merge(video_stat, on="video_id", how="left", suffixes=("", "_vstat"))
    return df


def verify_alignment(data_dir: str) -> dict:
    """
    Cross-check our pandas splits against the official numpy loader: same row
    counts, same (user_id, video_id, label) sequence. If this fails, every
    submission we produce is misaligned and every score is meaningless.
    """
    from official.data import load as official_load

    official = official_load(data_dir)
    ours = split_by_date(load_all_logs(data_dir))
    report = {}
    for name in SPLITS:
        off, our = official[name], ours[name]
        ok_len = len(off) == len(our)
        ok_seq = ok_len and all(
            str(o[1]) == str(u) and str(o[2]) == str(v) and o[6] == int(y)
            for o, u, v, y in zip(
                off, our["user_id"], our["video_id"], our[LABEL_COL].fillna(0).astype(int) != 0
            )
        )
        report[name] = {
            "official_rows": len(off), "our_rows": len(our),
            "lengths_match": ok_len, "sequence_matches": bool(ok_seq),
        }
    return report


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/KuaiRand-Pure/data")
    p.add_argument("--verify", action="store_true",
                   help="cross-check split alignment against official/data.py")
    a = p.parse_args()

    bundle = load_kuairand_pure(a.data_dir, load_features=False)
    for name, df in (("train", bundle.train), ("valid", bundle.val), ("test", bundle.test)):
        print(f"{name:5s} rows={len(df):>9,}  "
              f"{LABEL_COL} positive rate={df[LABEL_COL].mean():.4f}")
    if a.verify:
        print(json.dumps(verify_alignment(a.data_dir), indent=2))
