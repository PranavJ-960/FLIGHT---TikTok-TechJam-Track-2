"""Sequence-aware data loader: for each impression, the user's chronological
history of prior long_view=1 videos (DIN-style behavior sequence), most recent
last, capped at max_len. Reuses starter_kit's feature vocab/bucketing
conventions for apples-to-apples encoding; starter_kit/ itself is untouched.
"""
import csv
import os

import numpy as np

from starter_kit.data import FIELDS, SPLITS, _bucket_edges

LABEL = 'long_view'


def load_seq_rows(data_dir):
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
                             int(r['time_ms'])))
    return rows  # (date, user, video, author, tab, dur_ms, label, time_ms)


def build_history(rows, max_len=20):
    """History uses ALL rows (train+valid+test dates) so a valid/test impression can
    see the user's true past behavior, including train-period events — this is not
    leakage: only strictly-earlier-in-time long_view=1 events are used, never future
    ones, and no future label is read. Returns a list of video_id-string lists,
    aligned 1:1 with `rows`, each already time-truncated to <= max_len.
    """
    by_user = {}
    for i, x in enumerate(rows):
        by_user.setdefault(x[1], []).append(i)

    hist = [None] * len(rows)
    for idxs in by_user.values():
        idxs.sort(key=lambda i: rows[i][7])   # time_ms, ascending
        seen = []                              # (time_ms, video_id) of long_view=1 so far
        for i in idxs:
            hist[i] = [v for _, v in seen[-max_len:]]
            if rows[i][6] == 1:
                seen.append((rows[i][7], rows[i][2]))
    return hist


def split_rows(rows):
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def encode_seq(data_dir, max_len=20):
    rows = load_seq_rows(data_dir)
    hist = build_history(rows, max_len=max_len)
    by_date = {}
    for name, (lo, hi) in SPLITS.items():
        by_date[name] = [(i, x) for i, x in enumerate(rows) if lo <= x[0] <= hi]

    tr_rows = [x for _, x in by_date['train']]
    edges = _bucket_edges([x[5] for x in tr_rows])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr_rows:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))
    pad_idx = dim          # dedicated PAD embedding row, appended beyond the discrete vocab
    video_field = FIELDS.index('video_id')
    video_vocab, video_unk, video_offset = vocabs[video_field], unk[video_field], offsets[video_field]

    def vid_slot(v):
        return video_vocab.get(v, video_unk) + video_offset

    enc = {}
    for name, pairs in by_date.items():
        n = len(pairs)
        X = np.empty((n, len(FIELDS)), dtype=np.int32)
        H = np.full((n, max_len), pad_idx, dtype=np.int32)
        M = np.zeros((n, max_len), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        users = []
        for row_n, (orig_i, x) in enumerate(pairs):
            for f, v in enumerate(raw(x)):
                X[row_n, f] = vocabs[f].get(v, unk[f]) + offsets[f]
            y[row_n] = x[6]
            users.append(x[1])
            h = hist[orig_i]
            L = len(h)
            if L:
                H[row_n, max_len - L:] = [vid_slot(v) for v in h]
                M[row_n, max_len - L:] = 1.0
        enc[name] = (X, H, M, y, users)
    return enc, dim, pad_idx
