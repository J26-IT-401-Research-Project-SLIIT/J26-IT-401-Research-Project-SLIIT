"""
parkinson.py
------------
Direct standalone Python conversion of Parkinson.ipynb.
Contains the complete end-to-end pipeline:
1. Acoustic feature extraction (MFCCs, spectral, pitch, Praat jitter/shimmer/HNR)
2. Audio SHA-256 deduplication & dataset hygiene
3. Stratified train/test splitting (80/20)
4. Pipeline definitions & training for SVM, Random Forest, and MLP Neural Network
5. Performance evaluation, metrics reporting, and .joblib model serialization
"""

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Audio libraries
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature Column Definitions
VOICE_QUALITY_FEATURES = ["Jitter_local", "Shimmer_local", "HNR_dB", "Voice_Ratio"]
PROSODIC_FEATURES = ["Pitch_mean_Hz", "Pitch_std_Hz", "Speech_Rate_proxy"]
SPECTRAL_FEATURES = ["Spectral_Centroid", "Spectral_Bandwidth", "Zero_Crossing_Rate"]
MFCC_FEATURES = [f"MFCC{i}" for i in range(13)]

FEATURE_COLUMNS = [
    "duration_sec",
    "Pitch_mean_Hz",
    "Pitch_std_Hz",
    "Jitter_local",
    "Shimmer_local",
    "HNR_dB",
    "Voice_Ratio",
    "Speech_Rate_proxy",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Zero_Crossing_Rate",
] + MFCC_FEATURES


# ---------------------------------------------------------------------------
# 1. Feature Extraction
# ---------------------------------------------------------------------------
def extract_all_features(
    audio_path_or_bytes: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts 23 acoustic features from an audio recording.
    Matches the exact implementation from Cell 26 in Parkinson.ipynb.
    """
    if isinstance(audio_path_or_bytes, (bytes, io.BytesIO)):
        buf = io.BytesIO(audio_path_or_bytes) if isinstance(audio_path_or_bytes, bytes) else audio_path_or_bytes
        buf.seek(0)
        y, sr = sf.read(buf, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        name = filename or "sample.wav"
    else:
        path = Path(audio_path_or_bytes)
        name = filename or path.name
        y, sr = librosa.load(str(path), sr=None, mono=True)

    duration = float(len(y) / sr)
    if duration < 0.5:
        raise ValueError(f"Audio too short ({duration:.2f}s). Minimum required duration is 0.5s.")

    # Original sound object for Praat voice-quality analysis
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # Trim silence for general acoustic features
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trim) < int(sr * 0.25):
        y_trim = y

    trimmed_duration = float(len(y_trim) / sr)

    # 1. MFCC Features (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    # 2. Spectral Features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_trim, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y_trim)))

    # 3. Pitch / F0 (Librosa pyin)
    pitch_mean = np.nan
    pitch_std = np.nan
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_trim, fmin=75, fmax=min(600, sr / 2 - 1), sr=sr
        )
        valid_f0 = f0[np.isfinite(f0)]
        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))
            pitch_std = float(np.std(valid_f0))
    except Exception:
        pass

    # 4. Praat Voice Quality Features
    jitter = np.nan
    shimmer = np.nan
    hnr = np.nan
    voice_ratio = np.nan

    try:
        pitch_obj = call(sound, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch_obj.selected_array["frequency"]
        voice_ratio = float(np.mean(pitch_values > 0))

        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        harm = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = float(call(harm, "Get mean", 0, 0))
    except Exception:
        pass

    # 5. Onset Rate Proxy
    onset_rate_proxy = np.nan
    try:
        onset_frames = librosa.onset.onset_detect(y=y_trim, sr=sr)
        if trimmed_duration > 0:
            onset_rate_proxy = float(len(onset_frames) / trimmed_duration)
    except Exception:
        pass

    # Target Status
    status = (
        1 if "parkinson" in str(name).lower()
        else 0 if "healthy" in str(name).lower()
        else np.nan
    )

    row: Dict[str, Any] = {
        "filename": name,
        "duration_sec": duration,
        "Pitch_mean_Hz": pitch_mean,
        "Pitch_std_Hz": pitch_std,
        "Jitter_local": jitter,
        "Shimmer_local": shimmer,
        "HNR_dB": hnr,
        "Voice_Ratio": voice_ratio,
        "Speech_Rate_proxy": onset_rate_proxy,
        "Spectral_Centroid": centroid,
        "Spectral_Bandwidth": bandwidth,
        "Zero_Crossing_Rate": zcr,
        "Status": status,
    }

    for i, value in enumerate(mfcc_means):
        row[f"MFCC{i}"] = float(value)

    return row


def extract_dataset(audio_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Extracts features for all WAV files in audio_dir and saves to output_csv."""
    audio_files = sorted(audio_dir.rglob("*.wav"))
    logger.info("Found %d audio files in %s", len(audio_files), audio_dir)

    rows = []
    errors = []
    for idx, path in enumerate(audio_files, start=1):
        try:
            rows.append(extract_all_features(path))
        except Exception as e:
            errors.append((path.name, str(e)))

        if idx % 50 == 0 or idx == len(audio_files):
            logger.info("Processed %d/%d", idx, len(audio_files))

    features_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_csv, index=False)
    logger.info("Features saved to: %s (Success: %d, Failed: %d)", output_csv, len(features_df), len(errors))
    return features_df


# ---------------------------------------------------------------------------
# 2. SHA-256 Deduplication
# ---------------------------------------------------------------------------
def deduplicate_features(
    csv_path: Path,
    audio_root: Optional[Path],
    output_csv: Path
) -> pd.DataFrame:
    """Deduplicates dataset based on audio SHA-256 hash (Cell 30)."""
    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].astype(str)

    unique_filenames = []
    if audio_root and audio_root.exists():
        logger.info("Deduplicating audio by SHA-256 hash in: %s", audio_root)
        audio_groups: Dict[str, List[str]] = {}
        for path in sorted(audio_root.rglob("*.wav")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            audio_groups.setdefault(digest, []).append(path.name)

        csv_names = set(df["filename"])
        for digest, filenames in audio_groups.items():
            valid_names = [f for f in filenames if f in csv_names]
            if not valid_names:
                continue
            group_rows = df[df["filename"].isin(valid_names)]
            if group_rows["Status"].nunique() != 1:
                raise ValueError(f"Conflicting labels in duplicate group: {valid_names}")
            unique_filenames.append(sorted(valid_names)[0])

        unique_df = df[df["filename"].isin(unique_filenames)].drop_duplicates(subset="filename").copy()
    else:
        logger.warning("Audio directory not found. Deduplicating based on identical feature rows.")
        feature_cols = [c for c in df.columns if c not in ["filename", "Status"]]
        unique_df = df.drop_duplicates(subset=feature_cols).copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    unique_df.to_csv(output_csv, index=False)
    logger.info("Deduplication: %d -> %d unique rows. Saved to: %s", len(df), len(unique_df), output_csv)
    return unique_df


# ---------------------------------------------------------------------------
# 3. Model Training & Comparison
# ---------------------------------------------------------------------------
def train_models(
    unique_csv_path: Path,
    output_dir: Path = Path("saved_models")
) -> pd.DataFrame:
    """
    Trains SVM, Random Forest, and MLP Neural Network on the unique dataset.
    Evaluates on 80/20 test split and saves models via joblib (Cell 31).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(unique_csv_path)

    # Prepare X and y
    cols_to_drop = [c for c in ["filename", "Status"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Align columns strictly
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan
    X = X[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["Status"].astype(int)

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,  # enabled for API confidence estimation
                random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42
            )),
        ]),
        "mlp_neural_network": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )),
        ]),
    }

    results = []

    for name, model in models.items():
        print("\n" + "=" * 45)
        print(f"Training: {name.upper()}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(
            y_test, predictions,
            target_names=["Healthy", "Parkinson's"],
            zero_division=0
        ))

        model_path = output_dir / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        logger.info("Saved model to %s", model_path)

    # Compare models
    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    print("\n" + "=" * 50)
    print("MODEL COMPARISON (Ranked by F1-Score)")
    print("=" * 50)
    print(results_df.to_string(index=False))

    results_csv = output_dir / "model_comparison_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info("Saved results summary to %s", results_csv)

    return results_df


if __name__ == "__main__":
    # Check if unique CSV exists, or locate audio folder
    unique_csv = Path("parkinsons_voice_features_unique.csv")
    raw_csv = Path("parkinsons_voice_features.csv")
    audio_dir = Path("../features/archive/Voice_Dataset")

    if not unique_csv.exists():
        if raw_csv.exists():
            deduplicate_features(raw_csv, audio_dir if audio_dir.exists() else None, unique_csv)
        elif audio_dir.exists():
            extract_dataset(audio_dir, raw_csv)
            deduplicate_features(raw_csv, audio_dir, unique_csv)
        else:
            print("To train, please provide 'parkinsons_voice_features_unique.csv' or audio directory.")

    if unique_csv.exists():
        train_models(unique_csv, output_dir=Path("saved_models"))
