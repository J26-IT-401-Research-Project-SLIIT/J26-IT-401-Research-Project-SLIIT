from pathlib import Path
import tempfile
import numpy as np
from pads_motor.data import diagnosis_to_binary, extract_dataset
from pads_motor.features import features_from_record
from pads_motor.evaluate import participant_split,aggregate_by_participant
from scripts.make_synthetic_demo import make


def test_labels():
    assert diagnosis_to_binary("Parkinson's")==1
    assert diagnosis_to_binary('Healthy')==0
    assert diagnosis_to_binary('Essential Tremor') is None
    assert diagnosis_to_binary('Atypical Parkinsonism') is None


def test_synthetic_loader():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);make(root,n=32)
        X,meta=extract_dataset(root)
        assert X.shape[1:] == (900,6)
        assert len(X)==len(meta)
        assert set(meta.label)=={0,1}
        assert meta.participant_id.nunique()==32
        assert len(X)==32*2*4  # three tasks, Relaxed has two 1024-sample chunks
        split=participant_split(meta)
        assert split.participant_id.nunique()==32
        assert set(split.split)=={'train','val','test'}
        row=features_from_record(X[0])
        assert len(row)>=12
        assert all(np.isfinite(list(row.values())))
        prob=np.where(meta.label.to_numpy()==1,.7,.3)
        assert len(aggregate_by_participant(meta,prob))==32
