"""
features.py
------------
Feature engineering for the KuaiRand-Pure click-prediction / ranking task.

This module is intentionally the main "extension point" for the agent's
iterate-on-the-pipeline step (Figure 1, stage "Engineer features"). Each
FeatureSet is a small, named, self-contained transform so that an iteration
loop can add/remove/replace them and log exactly what changed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

CATEGORICAL_USER_COLS = [
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
]

CATEGORICAL_VIDEO_COLS = ["video_type", "upload_type"]

RAW_NUMERIC_COLS = [
    # user
    "follow_user_num", "fans_user_num", "friend_user_num", "register_days",
    "is_lowactive_period", "is_live_streamer", "is_video_author",
    # video basic
    "video_duration", "server_width", "server_height",
    # video statistic (engagement priors)
    "show_cnt", "play_cnt", "complete_play_cnt", "valid_play_cnt",
    "long_time_play_cnt", "like_cnt", "comment_cnt", "follow_cnt",
    "share_cnt", "collect_cnt", "download_cnt", "report_cnt",
    # interaction context
    "tab", "is_rand",
]

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Ordinal-encode categorical columns in place (LightGBM handles ints fine
    and this keeps train/val/test consistent without needing a fitted
    encoder object to persist)."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def add_engagement_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived ratios that are cheap, monotone-informative signals for CTR."""
    df = df.copy()
    eps = 1e-6
    if {"play_cnt", "show_cnt"}.issubset(df.columns):
        df["play_rate"] = df["play_cnt"] / (df["show_cnt"] + eps)
    if {"like_cnt", "play_cnt"}.issubset(df.columns):
        df["like_rate_given_play"] = df["like_cnt"] / (df["play_cnt"] + eps)
    if {"complete_play_cnt", "play_cnt"}.issubset(df.columns):
        df["completion_rate"] = df["complete_play_cnt"] / (df["play_cnt"] + eps)
    if {"video_duration"}.issubset(df.columns):
        df["log_video_duration"] = np.log1p(df["video_duration"].clip(lower=0))
    return df


def add_user_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple transforms of user activity counters (log-scale heavy tails)."""
    df = df.copy()
    for col in ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day from hourmin, useful because CTR is time-of-day dependent."""
    df = df.copy()
    if "hourmin" in df.columns:
        df["hour"] = (df["hourmin"] // 100).clip(0, 23)
    return df


def fill_missing(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


# The default pipeline. An agent iteration can append/remove/replace entries.
DEFAULT_FEATURE_PIPELINE: list[FeatureFn] = [
    add_time_features,
    add_engagement_ratio_features,
    add_user_activity_features,
]


def build_features(
    df: pd.DataFrame,
    pipeline: list[FeatureFn] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply the feature pipeline and return (feature_df, feature_column_names).
    `pipeline` defaults to DEFAULT_FEATURE_PIPELINE; pass a custom list to
    let an iteration try a different feature set.
    """
    pipeline = pipeline if pipeline is not None else DEFAULT_FEATURE_PIPELINE

    out = df.copy()
    for fn in pipeline:
        out = fn(out)

    out = encode_categoricals(out, CATEGORICAL_USER_COLS + CATEGORICAL_VIDEO_COLS)

    engineered_cols = [
        c for c in out.columns
        if c.startswith(("log_", "play_rate", "like_rate", "completion_rate", "hour"))
    ]
    feature_cols = [
        c for c in RAW_NUMERIC_COLS + CATEGORICAL_USER_COLS + CATEGORICAL_VIDEO_COLS + engineered_cols
        if c in out.columns
    ]
    feature_cols = sorted(set(feature_cols))

    out = fill_missing(out, feature_cols)
    return out, feature_cols
