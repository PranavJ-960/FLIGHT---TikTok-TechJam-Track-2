"""
data_loader.py
--------------
Loads the KuaiRand-Pure dataset and produces the fixed train/validation/test
split required by the TechJam "Autonomous ML Research Agent for Recommender
Systems" challenge:

    log_standard_4_08_to_4_21_pure.csv               -> TRAIN
    log_standard_4_22_to_5_08_pure.csv  (first 50%)  -> VALIDATION
    log_standard_4_22_to_5_08_pure.csv  (last 50%)   -> TEST (hidden in
                                                         practice -- only
                                                         touch this once)

The 50/50 split of the second log file is done chronologically using the
`time_ms` column, per the challenge's "first 50% / last 50%" rule.

Feature files (user_features_pure.csv, video_features_basic_pure.csv,
video_features_statistic_pure.csv) are loaded and can be joined onto the
interaction logs on user_id / video_id.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG_TRAIN_FILE = "log_standard_4_08_to_4_21_pure.csv"
LOG_VALTEST_FILE = "log_standard_4_22_to_5_08_pure.csv"
LOG_RANDOM_FILE = "log_random_4_22_to_5_08_pure.csv"  # optional, for OPE work

USER_FEATURES_FILE = "user_features_pure.csv"
VIDEO_BASIC_FILE = "video_features_basic_pure.csv"
VIDEO_STAT_FILE = "video_features_statistic_pure.csv"

LABEL_COL = "is_click"


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


def load_raw_logs(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two standard-policy log files (train source + val/test source)."""
    train_log = _read_csv_safe(os.path.join(data_dir, LOG_TRAIN_FILE))
    valtest_log = _read_csv_safe(os.path.join(data_dir, LOG_VALTEST_FILE))
    return train_log, valtest_log


def split_val_test(valtest_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split the 4/22-5/08 standard log into:
      - first 50% (by time_ms)  -> validation
      - last 50%  (by time_ms)  -> test
    """
    df = valtest_log.sort_values("time_ms").reset_index(drop=True)
    midpoint = len(df) // 2
    val = df.iloc[:midpoint].reset_index(drop=True)
    test = df.iloc[midpoint:].reset_index(drop=True)
    return val, test


def load_feature_tables(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_features = _read_csv_safe(os.path.join(data_dir, USER_FEATURES_FILE))
    video_basic = _read_csv_safe(os.path.join(data_dir, VIDEO_BASIC_FILE))
    video_stat = _read_csv_safe(os.path.join(data_dir, VIDEO_STAT_FILE))
    return user_features, video_basic, video_stat


def load_kuairand_pure(data_dir: str, load_features: bool = True) -> KuaiRandData:
    """Main entry point: returns a KuaiRandData bundle with train/val/test splits."""
    logger.info("Loading raw logs from %s", data_dir)
    train_log, valtest_log = load_raw_logs(data_dir)
    val_log, test_log = split_val_test(valtest_log)

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_log), len(val_log), len(test_log),
    )

    user_features = video_basic = video_stat = None
    if load_features:
        user_features, video_basic, video_stat = load_feature_tables(data_dir)

    return KuaiRandData(
        train=train_log,
        val=val_log,
        test=test_log,
        user_features=user_features,
        video_basic=video_basic,
        video_stat=video_stat,
    )


def merge_features(
    log_df: pd.DataFrame,
    user_features: pd.DataFrame | None,
    video_basic: pd.DataFrame | None,
    video_stat: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join user/video side information onto an interaction log."""
    df = log_df.copy()
    if user_features is not None:
        df = df.merge(user_features, on="user_id", how="left", suffixes=("", "_user"))
    if video_basic is not None:
        df = df.merge(video_basic, on="video_id", how="left", suffixes=("", "_vbasic"))
    if video_stat is not None:
        df = df.merge(video_stat, on="video_id", how="left", suffixes=("", "_vstat"))
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/KuaiRand-Pure/data")
    args = parser.parse_args()

    bundle = load_kuairand_pure(args.data_dir)
    print(bundle.train.head())
    print(f"Positive rate (train, is_click): {bundle.train[LABEL_COL].mean():.4f}")
