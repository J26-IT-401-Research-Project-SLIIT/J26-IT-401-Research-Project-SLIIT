import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, accuracy_score)
from sklearn.model_selection import train_test_split


def participant_split(meta, seed=42, test_frac=.15, val_frac=.15):
    """Stratify on subjects, never on individual windows/wrists."""
    people = meta[["participant_id", "label"]].drop_duplicates()
    if people.participant_id.duplicated().any():
        raise ValueError("Inconsistent participant labels")
    trainval, test = train_test_split(people, test_size=test_frac, random_state=seed,
                                      stratify=people.label)
    rest_share = val_frac / (1 - test_frac)
    train, val = train_test_split(trainval, test_size=rest_share, random_state=seed,
                                  stratify=trainval.label)
    lookup = {str(pid): subset for subset, frame in
              (("train", train), ("val", val), ("test", test))
              for pid in frame.participant_id}
    out = meta[["participant_id", "label"]].drop_duplicates().copy()
    out["split"] = out.participant_id.map(lookup)
    assert out.split.notna().all()
    for part in ("train", "val", "test"):
        group = out[out.split == part]
        if group.label.nunique() < 2:
            raise ValueError(f"Split {part} missing a class; collect more participants")
    return out


def aggregate_by_participant(meta, proba):
    a = meta[["participant_id", "label"]].copy()
    a["probability_pd"] = np.asarray(proba).reshape(-1)
    return a.groupby(["participant_id", "label"], as_index=False)["probability_pd"].mean()


def metrics_subject(meta, proba):
    sub = aggregate_by_participant(meta, proba)
    y = sub.label.to_numpy(dtype=int)
    yp = (sub.probability_pd.to_numpy() >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0,1]).ravel()
    return {
        "participants": int(len(sub)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "accuracy": float(accuracy_score(y, yp)),
        "f1_pd": float(f1_score(y, yp, zero_division=0)),
        "sensitivity_pd": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_healthy": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(y, sub.probability_pd)) if len(set(y))==2 else None,
        "confusion_matrix_healthy_pd": [[int(tn),int(fp)],[int(fn),int(tp)]],
    }
