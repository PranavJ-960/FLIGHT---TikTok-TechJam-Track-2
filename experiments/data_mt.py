"""Multi-task data loader: adds auxiliary label columns (is_click/is_like/is_follow)
alongside long_view. starter_kit/data.py only extracts a single LABEL column, so this
re-reads the same raw CSVs directly rather than editing starter_kit. Feature vocab
building and bucketing mirror starter_kit.data.encode exactly for apples-to-apples
feature encoding.
"""
import csv
import os

import numpy as np

from starter_kit.data import FIELDS, SPLITS, _bucket_edges

LABEL = 'long_view'
AUX_TASKS = ['is_click', 'is_like', 'is_follow']


def load_mt(data_dir, aux_tasks=AUX_TASKS):
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                labels = {LABEL: 1 if r[LABEL] != '0' else 0}
                for t in aux_tasks:
                    labels[t] = 1 if r[t] != '0' else 0
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), labels))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def encode_mt(splits, aux_tasks=AUX_TASKS):
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    all_tasks = [LABEL] + list(aux_tasks)
    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        Y = {t: np.empty(len(rws), dtype=np.float32) for t in all_tasks}
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            for t in all_tasks:
                Y[t][n] = x[6][t]
            users.append(x[1])
        enc[name] = (X, Y, users)
    return enc, int(sum(field_dims))
