"""Causal (leakage-safe) feature engineering for the LightGBM ranker: time-decayed
target-encoded rates, recency, session position, and a windowed "trending" rate for
video, plus temporal context (hour-of-day, day-of-week). Every entity-level feature
is computed as a single pass over ALL rows in global chronological order, using only
strictly-earlier interactions for that entity (and for the smoothing prior itself,
and for the rolling "recent" window) — same causal-ordering discipline as
experiments/data_seq.py's history sequences: never leaks the current row's own label
or any future row. A row's own query-time metadata (its timestamp, hour, weekday) is
always known at prediction time and is not leakage.

Low-cardinality categoricals (author_id, tab, dur_bucket) are encoded for LightGBM's
native categorical handling; user_id/video_id are deliberately represented only via
their causal features here, not as raw high-cardinality categoricals — trees split
poorly on that many distinct ids, whereas FM's learned embeddings (used elsewhere in
this repo) are exactly the right tool for id-level collaborative filtering.
starter_kit/ is not modified.
"""
import csv
import datetime
import os

import numpy as np
import pandas as pd

from starter_kit.data import SPLITS, _bucket_edges

LABEL = 'long_view'
CAT_FIELDS = ['author_id', 'tab', 'dur_bucket']
RECENT_WINDOW = 20


def load_causal_rows(data_dir):
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             int(r['time_ms']), int(r['hourmin'])))
    return rows   # date,user,video,author,tab,dur_ms,label,time_ms,hourmin


def _dow(date_int):
    y, m, d = date_int // 10000, (date_int // 100) % 100, date_int % 100
    return datetime.date(y, m, d).weekday()


def _causal_rate(df, key_col, label_col, smooth, gmean_prior):
    grp = df.groupby(key_col)[label_col]
    prior_sum = grp.cumsum() - df[label_col]
    prior_count = grp.cumcount()
    rate = (prior_sum + smooth * gmean_prior) / (prior_count + smooth)
    return rate, prior_count


def build_causal_features(rows, smooth=5.0, recent_window=RECENT_WINDOW):
    """One causal pass in global time order (vectorized via pandas). Returns a dict
    of feature-name -> array, aligned 1:1 with `rows`.
    """
    df = pd.DataFrame(rows, columns=['date', 'user', 'video', 'author', 'tab',
                                      'dur_ms', 'label', 'time_ms', 'hourmin'])
    df['orig_idx'] = np.arange(len(df))
    df = df.sort_values('time_ms', kind='mergesort').reset_index(drop=True)

    cum_pos = df['label'].cumsum() - df['label']
    cum_n = pd.Series(np.arange(len(df)), index=df.index)
    gmean_prior = (cum_pos / cum_n.where(cum_n > 0, 1)).where(cum_n > 0, 0.5)

    u_rate, u_count = _causal_rate(df, 'user', 'label', smooth, gmean_prior)
    v_rate, v_count = _causal_rate(df, 'video', 'label', smooth, gmean_prior)
    a_rate, a_count = _causal_rate(df, 'author', 'label', smooth, gmean_prior)
    # per-user, per-tab rate: does this user engage differently across tabs/feeds
    ut_rate, ut_count = _causal_rate(df, ['user', 'tab'], 'label', smooth, gmean_prior)
    # per-video, per-tab rate: does this video perform differently depending on which
    # tab/feed surfaces it (mirrors user_tab_rate, other direction)
    vt_rate, vt_count = _causal_rate(df, ['video', 'tab'], 'label', smooth, gmean_prior)

    # recency: ms since this entity's previous interaction; NaN for the first occurrence
    # (left as missing — LightGBM handles NaN natively rather than needing a fake sentinel)
    u_since = df.groupby('user')['time_ms'].diff()
    v_since = df.groupby('video')['time_ms'].diff()
    a_since = df.groupby('author')['time_ms'].diff()

    # session position: how many interactions this user has already had today (causal, resets per date)
    u_today_count = df.groupby(['user', 'date']).cumcount()

    # video "trending" rate: mean label over the last `recent_window` PRIOR interactions
    # with this video (shift(1) excludes the current row) — deliberately unsmoothed vs.
    # v_rate's all-time Bayesian average, so the tree can see recent vs. long-run divergence
    shifted = df.groupby('video')['label'].shift(1)
    v_recent_rate = shifted.groupby(df['video']).rolling(recent_window, min_periods=1).mean() \
                            .reset_index(level=0, drop=True)
    v_recent_rate = v_recent_rate.fillna(gmean_prior)
    v_trend = v_recent_rate - v_rate

    # author "trending" rate: same windowed-recent-vs-all-time idea, for author instead of video
    a_shifted = df.groupby('author')['label'].shift(1)
    a_recent_rate = a_shifted.groupby(df['author']).rolling(recent_window, min_periods=1).mean() \
                              .reset_index(level=0, drop=True)
    a_recent_rate = a_recent_rate.fillna(gmean_prior)
    a_trend = a_recent_rate - a_rate

    out = pd.DataFrame({
        'orig_idx': df['orig_idx'],
        'user_rate': u_rate, 'video_rate': v_rate, 'author_rate': a_rate,
        'user_tab_rate': ut_rate, 'user_tab_count': ut_count,
        'video_tab_rate': vt_rate, 'video_tab_count': vt_count,
        'user_count': u_count, 'video_count': v_count, 'author_count': a_count,
        'user_since_ms': u_since, 'video_since_ms': v_since, 'author_since_ms': a_since,
        'user_today_count': u_today_count,
        'video_recent_rate': v_recent_rate, 'video_trend': v_trend,
        'author_recent_rate': a_recent_rate, 'author_trend': a_trend,
    })
    out = out.sort_values('orig_idx').reset_index(drop=True)

    feat = {c: out[c].to_numpy(dtype=np.float32) for c in out.columns if c != 'orig_idx'}
    feat['hour'] = np.array([x[8] // 100 for x in rows], dtype=np.float32)
    feat['dow'] = np.array([_dow(x[0]) for x in rows], dtype=np.int32)
    return feat


# video_tab_rate/count and author_recent_rate/trend were tested (see build_causal_features)
# and REJECTED: tuned LightGBM on the 24-feature set (0.6293) came in below the proven
# 18-feature set (0.6306) — likely redundant with user_tab_rate/video_rate/author_rate,
# diluting feature_fraction sampling (same failure mode as the FM-embedding-stacking
# rejection). Left computed above for the record, excluded from the active feature set.
NUMERIC_FIELDS = ['user_rate', 'video_rate', 'author_rate', 'user_tab_rate', 'user_tab_count',
                   'user_count', 'video_count', 'author_count',
                   'user_since_ms', 'video_since_ms', 'author_since_ms', 'user_today_count',
                   'video_recent_rate', 'video_trend', 'duration_ms', 'hour']

# within-user RELATIVE features: everything above is an absolute, per-entity statistic.
# These instead answer "how does this candidate compare to the OTHER candidates this same
# user was shown" — percentile rank within (user, split), computed only from already-causal
# feature VALUES (never a sibling row's own label — that would be real leakage, since GAUC/
# nDCG@5 evaluate a user's whole split-group at once, so using another row's *label* from
# the same group is using information "from the answer key" of a simultaneously-scored row;
# using another row's already-causal *feature* is not, since those don't depend on any
# row's own label either). Only candidate-varying signals are included — ranking a pure
# per-user running stat (user_rate, user_count, ...) within that same user's group would
# just recover temporal order, not real comparative signal.
RELATIVE_SOURCE_COLS = ['video_rate', 'author_rate', 'video_recent_rate', 'video_trend', 'duration_ms']


def encode_causal(data_dir, smooth=5.0, recent_window=RECENT_WINDOW, with_relative=False):
    """Fully vectorized (no per-row Python loop over numpy arrays — that pattern is
    catastrophically slow at this row count; see [[project-gpu-parallel-infra]] memory).
    with_relative=True adds the within-user relative/percentile-rank features (see
    RELATIVE_SOURCE_COLS) — off by default so existing callers get the proven feature set
    unchanged.
    """
    rows = load_causal_rows(data_dir)
    feat = build_causal_features(rows, smooth=smooth, recent_window=recent_window)

    dates = np.array([x[0] for x in rows], dtype=np.int64)
    durations = np.array([x[5] for x in rows], dtype=np.float32)
    labels = np.array([x[6] for x in rows], dtype=np.float32)
    users_all = np.array([x[1] for x in rows], dtype=object)
    authors_all = np.array([x[3] for x in rows], dtype=object)
    tabs_all = np.array([x[4] for x in rows], dtype=object)

    lo_tr, hi_tr = SPLITS['train']
    train_mask = (dates >= lo_tr) & (dates <= hi_tr)
    edges = _bucket_edges(durations[train_mask].tolist())
    dur_bucket_all = np.searchsorted(edges, durations).astype(np.int32)

    author_vocab = {v: i for i, v in enumerate(pd.unique(authors_all[train_mask]))}
    tab_vocab = {v: i for i, v in enumerate(pd.unique(tabs_all[train_mask]))}
    author_unk, tab_unk = len(author_vocab), len(tab_vocab)
    author_idx_all = pd.Series(authors_all).map(author_vocab).fillna(author_unk).to_numpy(dtype=np.int32)
    tab_idx_all = pd.Series(tabs_all).map(tab_vocab).fillna(tab_unk).to_numpy(dtype=np.int32)

    causal_cols = [f for f in NUMERIC_FIELDS if f not in ('duration_ms', 'hour')]

    enc = {}
    for name, (lo, hi) in SPLITS.items():
        mask = (dates >= lo) & (dates <= hi)
        idx = np.nonzero(mask)[0]
        num = np.column_stack([feat[f][idx] for f in causal_cols] +
                               [durations[idx], feat['hour'][idx]]).astype(np.float32)
        dow_col = feat['dow'][idx]
        cat = np.column_stack([author_idx_all[idx], tab_idx_all[idx], dur_bucket_all[idx]]).astype(np.int32)
        if with_relative:
            split_users = users_all[idx]
            rel_df = pd.DataFrame({c: num[:, causal_cols.index(c)] if c in causal_cols else durations[idx]
                                    for c in RELATIVE_SOURCE_COLS})
            rel_df['user'] = split_users
            rel_cols = []
            for c in RELATIVE_SOURCE_COLS:
                rel_cols.append(rel_df.groupby('user')[c].rank(pct=True).to_numpy(dtype=np.float32))
            num = np.column_stack([num] + rel_cols)
        enc[name] = {'num': num, 'dow': dow_col, 'cat': cat,
                      'y': labels[idx], 'users': users_all[idx].tolist()}
    feature_names = NUMERIC_FIELDS + ['dow'] + CAT_FIELDS
    if with_relative:
        feature_names = NUMERIC_FIELDS + [f'{c}_rank_in_user' for c in RELATIVE_SOURCE_COLS] + ['dow'] + CAT_FIELDS
    return enc, feature_names
