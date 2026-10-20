"""
feature_extractor.py
--------------------
Audio feature extraction module for Parkinson's disease detection.
Extracts voice quality, prosodic, spectral, and MFCC features using
librosa and Praat (parselmouth).
"""

import io
import math
import logging
from pathlib import Path
from typing import Union, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature Column Definitions (Order must be strictly preserved)
VOICE_QUALITY_FEATURES = [
    "Jitter_local",
    "Shimmer_local",
    "HNR_dB",
    "Voice_Ratio",
]

PROSODIC_FEATURES = [
    "Pitch_mean_Hz",
    "Pitch_std_Hz",
    "Speech_Rate_proxy",
]

SPECTRAL_FEATURES = [
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Zero_Crossing_Rate",
]

MFCC_FEATURES = [f"MFCC{i}" for i in range(13)]

# Exactly 23 features used for model input
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


def load_audio_signal(
    audio_input: Union[str, Path, bytes, io.BytesIO],
    sr: Optional[int] = None
) -> tuple[np.ndarray, int]:
    """
    Loads an audio signal from a filepath, raw bytes, or BytesIO buffer.
    Always returns a 1D mono float numpy array and sampling rate.
    """
    if isinstance(audio_input, (bytes, io.BytesIO)):
        buffer = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else audio_input
        buffer.seek(0)
        y, sample_rate = sf.read(buffer, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr is not None and sample_rate != sr:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
        return y, sample_rate

    # If it is a file path
    path = Path(audio_input)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, sample_rate = librosa.load(str(path), sr=sr, mono=True)
    return y, sample_rate


def extract_features_from_audio(
    audio_input: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts all 23 acoustic features from an audio recording.

    Parameters:
        audio_input: File path (str/Path) or in-memory audio (bytes/BytesIO)
        filename: Optional filename identifier

    Returns:
        dict containing:
            - 'filename': str
            - 'duration_sec': float
            - Pitch, Jitter, Shimmer, HNR, Voice Ratio
            - Speech Rate proxy
            - Spectral Centroid, Bandwidth, ZCR
            - MFCC0 to MFCC12
            - 'Status': Optional target (1 for Parkinson's, 0 for Healthy, or NaN if unlabelled)
    """
    # Determine name
    if filename is None:
        if isinstance(audio_input, (str, Path)):
            filename = Path(audio_input).name
        else:
            filename = "audio_sample.wav"

    y, sr = load_audio_signal(audio_input)
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
            y_trim,
            fmin=75,
            fmax=min(600, sr / 2 - 1),
            sr=sr
        )
        valid_f0 = f0[np.isfinite(f0)]
        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))
            pitch_std = float(np.std(valid_f0))
    except Exception as e:
        logger.debug("Failed extracting librosa pitch: %s", e)

    # 4. Praat Voice Quality Features (Jitter, Shimmer, HNR, Voice Ratio)
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
    except Exception as e:
        logger.debug("Failed extracting Praat voice quality features: %s", e)

    # 5. Speech Rate Proxy (Onset rate proxy)
    onset_rate_proxy = np.nan
    try:
        onset_frames = librosa.onset.onset_detect(y=y_trim, sr=sr)
        if trimmed_duration > 0:
            onset_rate_proxy = float(len(onset_frames) / trimmed_duration)
    except Exception as e:
        logger.debug("Failed extracting onset rate proxy: %s", e)

    # Infer status from filename/path if available
    fn_lower = str(filename).lower()
    if "parkinson" in fn_lower:
        status = 1
    elif "healthy" in fn_lower:
        status = 0
    else:
        status = np.nan

    row: Dict[str, Any] = {
        "filename": str(filename),
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

    for i, val in enumerate(mfcc_means):
        row[f"MFCC{i}"] = float(val)

    return row


def extract_features_to_dataframe(
    audio_dir_or_paths: Union[str, Path, List[Path]],
    output_csv: Optional[Union[str, Path]] = None,
    progress_step: int = 50
) -> pd.DataFrame:
    """
    Extracts features for all WAV files in a directory or list of file paths.
    Optionally saves the resulting DataFrame to CSV.
    """
    if isinstance(audio_dir_or_paths, (str, Path)):
        audio_dir = Path(audio_dir_or_paths)
        audio_files = sorted(audio_dir.rglob("*.wav"))
    else:
        audio_files = sorted(audio_dir_or_paths)

    logger.info("Found %d audio files to process.", len(audio_files))

    rows: List[Dict[str, Any]] = []
    errors: List[tuple[str, str]] = []

    for idx, path in enumerate(audio_files, start=1):
        try:
            row = extract_features_from_audio(path, filename=path.name)
            rows.append(row)
        except Exception as e:
            errors.append((path.name, str(e)))

        if idx % progress_step == 0 or idx == len(audio_files):
            logger.info("Processed %d/%d audio files", idx, len(audio_files))

    df = pd.DataFrame(rows)
    logger.info("Extraction complete. Successful: %d, Failed: %d", len(df), len(errors))

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Features saved to: %s", out_path)

    return df
