"""
predictor.py
------------
Inference engine and model serving interface.
Designed specifically for easy integration with FastAPI and other web backends.
Accepts audio file paths, raw bytes, or precomputed feature dictionaries,
extracts acoustic features, runs pipeline inference, and returns structured JSON-ready results.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from feature_extractor import (
    FEATURE_COLUMNS,
    VOICE_QUALITY_FEATURES,
    PROSODIC_FEATURES,
    SPECTRAL_FEATURES,
    MFCC_FEATURES,
    extract_features_from_audio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ParkinsonPredictor:
    """
    Production-ready inference class for Parkinson's disease audio classification.
    """

    AVAILABLE_MODELS = ["random_forest", "svm", "mlp_neural_network"]

    def __init__(
        self,
        model_name: str = "random_forest",
        models_dir: Union[str, Path] = "saved_models",
        default_fallback: bool = True
    ):
        """
        Initializes predictor and loads specified trained model pipeline.

        Parameters:
            model_name: One of 'random_forest', 'svm', 'mlp_neural_network'
            models_dir: Directory containing saved .joblib model files
            default_fallback: If requested model is not found, attempt to load any available model
        """
        self.models_dir = Path(models_dir)
        self.model_name = model_name.lower().replace(" ", "_")
        self.pipeline: Optional[Pipeline] = None
        self._load_model(default_fallback=default_fallback)

    def _resolve_model_path(self, name: str) -> Optional[Path]:
        candidates = [
            self.models_dir / f"{name}_model.joblib",
            self.models_dir / f"{name}.joblib",
            Path(f"saved_models/{name}_model.joblib"),
            Path(f"../Saved Models/{name}_model.joblib"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_model(self, default_fallback: bool = True) -> None:
        target_path = self._resolve_model_path(self.model_name)

        if target_path and target_path.exists():
            logger.info("Loading model artifact from: %s", target_path)
            self.pipeline = joblib.load(target_path)
            return

        if default_fallback:
            # Check if any model exists in directory
            for alt in self.AVAILABLE_MODELS:
                alt_path = self._resolve_model_path(alt)
                if alt_path and alt_path.exists():
                    logger.warning(
                        "Model '%s' not found. Falling back to '%s' at %s",
                        self.model_name, alt, alt_path
                    )
                    self.model_name = alt
                    self.pipeline = joblib.load(alt_path)
                    return

        logger.warning(
            "No saved model found in '%s'. You must run training before making predictions.",
            self.models_dir
        )
        self.pipeline = None

    def predict_features(
        self,
        features_dict: Dict[str, Any],
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs prediction on a dictionary of acoustic features.
        """
        pipeline = self.pipeline
        if model_override:
            alt_path = self._resolve_model_path(model_override)
            if alt_path and alt_path.exists():
                pipeline = joblib.load(alt_path)

        if pipeline is None:
            raise RuntimeError(
                "Model pipeline is not loaded. Ensure trained models exist in saved_models directory."
            )

        # Prepare single-row DataFrame aligned strictly to FEATURE_COLUMNS
        row = {}
        for col in FEATURE_COLUMNS:
            row[col] = [features_dict.get(col, np.nan)]

        df_input = pd.DataFrame(row)
        df_input = df_input.replace([np.inf, -np.inf], np.nan)

        pred_class = int(pipeline.predict(df_input)[0])

        probabilities = {}
        confidence = 0.5
        if hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(df_input)[0]
                probabilities = {
                    "Healthy": round(float(probs[0]), 4),
                    "Parkinson's": round(float(probs[1]), 4),
                }
                confidence = float(probs[pred_class])
            except Exception as e:
                logger.debug("predict_proba failed: %s", e)

        # Group features for human-readable API response
        grouped_features = {
            "voice_quality": {k: features_dict.get(k) for k in VOICE_QUALITY_FEATURES if k in features_dict},
            "prosodic": {k: features_dict.get(k) for k in PROSODIC_FEATURES if k in features_dict},
            "spectral": {k: features_dict.get(k) for k in SPECTRAL_FEATURES if k in features_dict},
            "mfcc": {k: features_dict.get(k) for k in MFCC_FEATURES if k in features_dict},
        }

        return {
            "prediction": "Parkinson's Disease Detected" if pred_class == 1 else "Healthy Voice",
            "status": pred_class,
            "confidence_score": round(confidence, 4),
            "probabilities": probabilities,
            "model_used": model_override or self.model_name,
            "feature_groups": grouped_features,
            "all_features": {k: features_dict.get(k) for k in FEATURE_COLUMNS if k in features_dict},
        }

    def predict_audio(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO],
        filename: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end inference from an audio file path or audio byte stream.

        Parameters:
            audio_input: Path to audio file or raw bytes/BytesIO from file upload
            filename: Optional original audio filename
            model_override: Optional alternative model to use for this inference

        Returns:
            Dict containing prediction, status code, confidence, and extracted acoustic features.
        """
        extracted = extract_features_from_audio(audio_input, filename=filename)
        result = self.predict_features(extracted, model_override=model_override)
        result["filename"] = extracted.get("filename")
        result["duration_sec"] = extracted.get("duration_sec")
        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predictor.py <path_to_audio.wav> [model_name]")
        print("Available models: random_forest (default), svm, mlp_neural_network")
        sys.exit(1)

    audio_file = sys.argv[1]
    chosen_model = sys.argv[2] if len(sys.argv) > 2 else "random_forest"

    pred_service = ParkinsonPredictor(model_name=chosen_model)
    output = pred_service.predict_audio(audio_file)
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Audio File:     {output.get('filename')}")
    print(f"Duration:       {output.get('duration_sec', 0):.2f} seconds")
    print(f"Model Used:     {output.get('model_used')}")
    print(f"Diagnosis:      {output.get('prediction')}")
    print(f"Confidence:     {output.get('confidence_score', 0) * 100:.2f}%")
    print(f"Probabilities:  {output.get('probabilities')}")
    print("=" * 50)

