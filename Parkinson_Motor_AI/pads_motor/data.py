"""Load ORIGINAL PADS 1.0.0 movement JSON and timeseries/*.txt.

Source: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
Accelerometer columns are in g; gyro in rad/s. No 0-4 scores are provided.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

TASKS_DEFAULT = ("Relaxed", "RelaxedTask", "StretchHold")
ACC = ("Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z")
GYR = ("Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z")


def diagnosis_to_binary(condition: str):
    """Strict classification: DD is NOT healthy; exclude all other conditions."""
    c = str(condition).strip().lower().replace("’", "'")
    if c in ("healthy", "healthy control", "healthy controls"):
        return 0
    if c in ("parkinson's", "parkinson's disease", "parkinsons", "parkinson disease"):
        return 1
    return None


def load_patient_map(root: Path):
    folder = root / "patients"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing {folder}; unzip official PADS dataset first")
    by_id, skipped = {}, {}
    for f in sorted(folder.glob("patient_*.json")):
        obj = json.loads(f.read_text(encoding="utf-8"))
        pid = str(obj["id"]).zfill(3)
        label = diagnosis_to_binary(obj.get("condition", ""))
        if label is not None:
            by_id[pid] = (label, obj.get("condition", ""))
        else:
            condition = str(obj.get("condition", "<missing>"))
            skipped[condition] = skipped.get(condition, 0) + 1
    if not by_id:
        raise ValueError("No eligible PD/Healthy labels detected: check patients/patient_*.json")
    return by_id, skipped


def _raw_segments(path: Path, channels: list[str], samples_per_chunk=1024):
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if arr.shape[1] != len(channels):
        raise ValueError(f"{path.name}: expected {len(channels)} CSV columns {channels}, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path.name} contains missing/non-finite signal values")
    pos = {name: i for i, name in enumerate(channels)}
    missing = set(ACC + GYR) - set(pos)
    if missing:
        raise ValueError(f"{path.name}: channels missing: {sorted(missing)}")
    indices = [pos[ch] for ch in ACC + GYR]
    if arr.shape[0] < samples_per_chunk:
        return
    # Divide official 2048-sample sessions into two 1024-sample records.
    for j in range(arr.shape[0] // samples_per_chunk):
        block = arr[j * samples_per_chunk:(j + 1) * samples_per_chunk]
        yield j, block[:, indices]


def extract_dataset(root: Path, tasks=TASKS_DEFAULT, trim_first=50, length=900,
                    max_participants=None, strict=False):
    """Return features input tensor and metadata, one row per selected wrist/task/window.

    50 samples removed (0.5 s startup vibration). 900 remaining samples (9 s)
    fit both the nominal 1024-sample and 2048-sample task recordings.
    """
    root = Path(root)
    people, excluded = load_patient_map(root)
    print(f"Participant labels: {len(people)} PD/Healthy, excluded groups: {excluded}")
    observations = root / "movement"
    if not observations.is_dir():
        raise FileNotFoundError(f"Missing {observations}")
    if trim_first + length > 1024:
        raise ValueError("trim_first + length must be <= 1024 for the standard PADS task chunk")
    signals, metadata, problems = [], [], []
    eligible = sorted(people)
    if max_participants is not None:
        eligible = eligible[:max_participants]  # debugging ONLY: not a representative subset
    for pid in eligible:
        obs_path = observations / f"observation_{pid}.json"
        if not obs_path.exists():
            problems.append(f"missing observation: {obs_path.name}")
            continue
        obs = json.loads(obs_path.read_text(encoding="utf-8"))
        for session in obs.get("session", []):
            task = session.get("record_name", "")
            if tasks and task not in tasks:
                continue
            for rec in session.get("records", []):
                name = rec.get("file_name", "")
                path = observations / name
                if path.suffix.lower() != ".txt":
                    continue  # only official original TXT format, not binary preprocessed files
                if not path.exists():
                    problems.append(f"missing timeseries: {name}")
                    continue
                channels = rec.get("channels", [])
                units = rec.get("units", [])
                if len(channels) != len(units):
                    problems.append(f"missing channel unit data: {name}")
                    continue
                acc_units = [units[channels.index(k)] for k in ACC if k in channels]
                gyr_units = [units[channels.index(k)] for k in GYR if k in channels]
                if acc_units != ["g"] * 3 or gyr_units != ["rad/s"] * 3:
                    problems.append(f"unexpected units: {name} = {acc_units} / {gyr_units}")
                    continue
                try:
                    for j, block in _raw_segments(path, channels):
                        signal = block[trim_first:trim_first + length].copy()
                        if len(signal) != length:
                            continue
                        signals.append(signal)
                        metadata.append({
                            "participant_id": pid,
                            "label": people[pid][0],
                            "condition": people[pid][1],
                            "task": task,
                            "wrist": rec.get("device_location", "Unknown"),
                            "segment": j + 1,
                            "source": str(path.relative_to(root)),
                        })
                except (ValueError, OSError) as exc:
                    problems.append(f"{name}: {exc}")
    if problems:
        print(f"Skipped {len(problems)} missing/invalid records. First 5: {problems[:5]}")
        if strict:
            raise ValueError(f"Strict mode: {len(problems)} missing/invalid records")
    if not signals:
        raise ValueError("No recordings extracted. Check --root and chosen --tasks; ensure movement/timeseries .txt exists")
    meta = pd.DataFrame(metadata)
    series = np.stack(signals).astype(np.float32)
    print(f"Extracted {len(meta)} records from {meta.participant_id.nunique()} participants; tensor {series.shape}")
    return series, meta
