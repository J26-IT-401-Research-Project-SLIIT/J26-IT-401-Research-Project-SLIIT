"""
dataset.py
----------
Dataset management, deduplication, and data preprocessing module.
Handles loading feature CSVs, deduplicating identical audio recordings using
SHA-256 content hashing, and preparing training splits.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd

from feature_extractor import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def deduplicate_dataset(
    csv_path: Union[str, Path],
    audio_root: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Deduplicates the dataset based on audio content hash (SHA-256) or feature rows.
    Guarantees that duplicate recordings of the same voice test are not represented twice.

    Parameters:
        csv_path: Path to extracted features CSV (e.g. parkinsons_voice_features.csv)
        audio_root: Path to the root audio directory (e.g. Voice_Dataset)
        output_csv: Destination path for unique features CSV

    Returns:
        pd.DataFrame containing unique records
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Features file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df["filename"] = df["filename"].astype(str)
    initial_count = len(df)
    logger.info("Loaded dataset with %d rows from %s", initial_count, csv_file)

    audio_dir = Path(audio_root) if audio_root else None

    if audio_dir and audio_dir.exists():
        logger.info("Computing SHA-256 hashes for audio files in: %s", audio_dir)
        audio_groups: dict[str, List[str]] = {}
        for path in sorted(audio_dir.rglob("*.wav")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            audio_groups.setdefault(digest, []).append(path.name)

        unique_filenames = []
        csv_filenames_set = set(df["filename"])

        for digest, filenames in audio_groups.items():
            filenames_in_csv = [name for name in filenames if name in csv_filenames_set]
            if not filenames_in_csv:
                continue

            group_rows = df[df["filename"].isin(filenames_in_csv)]
            if group_rows["Status"].nunique() > 1:
                raise ValueError(
                    f"Conflicting target labels in duplicate group: {filenames_in_csv}"
                )

            # Deterministically keep first alphabetical file in duplicate group
            unique_filenames.append(sorted(filenames_in_csv)[0])

        unique_df = (
            df[df["filename"].isin(unique_filenames)]
            .drop_duplicates(subset="filename")
            .copy()
        )
    else:
        logger.warning(
            "Audio root folder not provided or does not exist. Falling back to feature-based deduplication."
        )
        feature_cols = [c for c in df.columns if c not in ["filename", "Status"]]
        unique_df = df.drop_duplicates(subset=feature_cols).copy()

    logger.info("Deduplication complete: %d -> %d rows", initial_count, len(unique_df))

    if output_csv:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        unique_df.to_csv(out_p, index=False)
        logger.info("Saved unique dataset to: %s", out_p)

    return unique_df


def load_dataset_for_training(
    csv_path: Union[str, Path]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads features CSV and prepares feature matrix X and target series y.
    Replaces +/- inf with NaN, validates presence of required columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {path}")

    df = pd.read_csv(path)
    if "Status" not in df.columns:
        raise ValueError("Missing 'Status' target column in dataset.")

    # Drop non-feature columns
    cols_to_drop = [c for c in ["filename", "Status"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Ensure all required features are present
    missing_cols = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing_cols:
        logger.warning("Dataset is missing columns %s. Filling with NaN.", missing_cols)
        for col in missing_cols:
            X[col] = np.nan

    # Order columns strictly according to FEATURE_COLUMNS
    X = X[FEATURE_COLUMNS].copy()

    # Clean infinities
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["Status"].astype(int)

    return X, y
