"""
pipeline.py
-----------
All-in-one command-line interface for the Parkinson's Voice AI project.
Enables running individual phases (extraction, deduplication, training, inference)
or the complete end-to-end pipeline from the terminal.
"""

import argparse
import logging
import sys
from pathlib import Path

from dataset import deduplicate_dataset
from feature_extractor import extract_features_to_dataframe
from predictor import ParkinsonPredictor
from train import evaluate_feature_groups_cv, train_and_evaluate_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Parkinson's Voice ML Pipeline (Feature Extraction -> Deduplication -> Training -> Inference)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # 1. Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract acoustic features from audio folder")
    extract_parser.add_argument("--audio-dir", type=str, required=True, help="Path to audio files directory")
    extract_parser.add_argument("--output", type=str, default="parkinsons_voice_features.csv", help="Output CSV path")

    # 2. Deduplicate command
    dedup_parser = subparsers.add_parser("deduplicate", help="Deduplicate features using audio SHA-256 hashes")
    dedup_parser.add_argument("--csv", type=str, default="parkinsons_voice_features.csv", help="Source features CSV")
    dedup_parser.add_argument("--audio-dir", type=str, default=None, help="Root folder of audio files")
    dedup_parser.add_argument("--output", type=str, default="parkinsons_voice_features_unique.csv", help="Output unique CSV")

    # 3. Train command
    train_parser = subparsers.add_parser("train", help="Train and compare SVM, Random Forest, and MLP models")
    train_parser.add_argument("--csv", type=str, default="parkinsons_voice_features_unique.csv", help="Features CSV")
    train_parser.add_argument("--output-dir", type=str, default="saved_models", help="Directory to save models")
    train_parser.add_argument("--run-cv", action="store_true", help="Also run 5-fold feature groups ablation")

    # 4. Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict Parkinson's status for an audio recording")
    predict_parser.add_argument("--audio", type=str, required=True, help="Path to test WAV audio file")
    predict_parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "svm", "mlp_neural_network"], help="Model to use")
    predict_parser.add_argument("--models-dir", type=str, default="saved_models", help="Directory containing models")

    # 5. Full pipeline command
    full_parser = subparsers.add_parser("all", help="Run full pipeline: extract -> deduplicate -> train")
    full_parser.add_argument("--audio-dir", type=str, required=True, help="Path to raw audio folder")
    full_parser.add_argument("--models-dir", type=str, default="saved_models", help="Output directory for models")

    args = parser.parse_args()

    if args.command == "extract":
        extract_features_to_dataframe(args.audio_dir, output_csv=args.output)

    elif args.command == "deduplicate":
        deduplicate_dataset(args.csv, audio_root=args.audio_dir, output_csv=args.output)

    elif args.command == "train":
        if args.run_cv:
            evaluate_feature_groups_cv(args.csv)
        train_and_evaluate_models(args.csv, output_dir=args.output_dir)

    elif args.command == "predict":
        predictor = ParkinsonPredictor(model_name=args.model, models_dir=args.models_dir)
        result = predictor.predict_audio(args.audio)
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"File:           {result['filename']}")
        print(f"Duration:       {result['duration_sec']:.2f} seconds")
        print(f"Model:          {result['model_used']}")
        print(f"Prediction:     {result['prediction']}")
        print(f"Confidence:     {result['confidence_score'] * 100:.2f}%")
        print(f"Probabilities:  {result['probabilities']}")
        print("=" * 50)

    elif args.command == "all":
        raw_csv = "parkinsons_voice_features.csv"
        unique_csv = "parkinsons_voice_features_unique.csv"
        logger.info("Step 1: Extracting features from %s...", args.audio_dir)
        extract_features_to_dataframe(args.audio_dir, output_csv=raw_csv)
        logger.info("Step 2: Deduplicating dataset...")
        deduplicate_dataset(raw_csv, audio_root=args.audio_dir, output_csv=unique_csv)
        logger.info("Step 3: Training models...")
        train_and_evaluate_models(unique_csv, output_dir=args.models_dir)
        logger.info("Pipeline execution completed successfully.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
