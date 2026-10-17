#!/usr/bin/env python3
"""Convert PADS original JSON + timeseries .txt to ready-for-ML tensors/features."""
import argparse
from pathlib import Path
import numpy as np
from pads_motor.data import extract_dataset, TASKS_DEFAULT
from pads_motor.features import build_features
from pads_motor.evaluate import participant_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/pads_dataset')
    p.add_argument('--out', default='outputs')
    p.add_argument('--tasks', default=','.join(TASKS_DEFAULT),
                   help='Comma-separated original PADS task names. Empty means all tasks.')
    p.add_argument('--max-participants', type=int, default=None, help='Debug only; biased if first N')
    p.add_argument('--strict', action='store_true')
    a = p.parse_args()
    tasks = tuple(s.strip() for s in a.tasks.split(',') if s.strip())
    X, meta = extract_dataset(Path(a.root),tasks=tasks,max_participants=a.max_participants,strict=a.strict)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out/'metadata.csv',index=False)
    np.save(out/'signals.npy', X, allow_pickle=False)
    features = build_features(X,meta)
    features.to_csv(out/'features.csv',index=False)
    splits = participant_split(meta)
    splits.to_csv(out/'participant_splits.csv',index=False)
    print('Wrote:',out/'signals.npy', out/'metadata.csv',out/'features.csv',out/'participant_splits.csv')
    print('Participants by split/label:\n',splits.groupby(['split','label']).size().to_string())

if __name__=='__main__':main()
