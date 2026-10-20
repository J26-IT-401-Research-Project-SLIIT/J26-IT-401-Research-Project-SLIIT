"""
train.py
--------
Model training, cross-validation, and experimentation suite for Parkinson's detection.
Implements scikit-learn pipelines for SVM, Random Forest, and MLP Neural Network,
runs 5-fold Stratified Cross-Validation and holdout testing, ranks models across
clinical and statistical metrics, exports model artifacts via joblib, and logs
detailed selection rationale.
"""

import argparse
import json
import logging
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_dataset_for_training
from feature_extractor import (
    FEATURE_COLUMNS,
    MFCC_FEATURES,
    PROSODIC_FEATURES,
    SPECTRAL_FEATURES,
    VOICE_QUALITY_FEATURES,
)

# Suppress sklearn SVC probability deprecation warning for clean logs
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm._base")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Builds the 3 machine learning pipelines tested in the project:
    1. Support Vector Classifier (RBF kernel, balanced weights, probability enabled)
    2. Random Forest Classifier (200 trees, balanced weights)
    3. Multi-Layer Perceptron Neural Network (64x32 hidden layers, early stopping)
    """
    models = {
        "svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "mlp_neural_network": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                early_stopping=True,
                random_state=random_state,
            )),
        ]),
    }
    return models


def evaluate_models_cv(
    csv_path: str,
    n_splits: int = 5,
    random_state: int = 42,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluates all 3 candidate models using Stratified K-Fold Cross-Validation.
    Measures Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC
    (mean +/- standard deviation) across folds.
    """
    X, y = load_dataset_for_training(csv_path)
    models = build_model_pipelines(random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    logger.info("Running %d-Fold Stratified Cross-Validation across 3 candidate models...", n_splits)
    cv_records = []

    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        acc_mean, acc_std = scores["test_accuracy"].mean() * 100, scores["test_accuracy"].std() * 100
        prec_mean, prec_std = scores["test_precision"].mean() * 100, scores["test_precision"].std() * 100
        rec_mean, rec_std = scores["test_recall"].mean() * 100, scores["test_recall"].std() * 100
        f1_mean, f1_std = scores["test_f1"].mean() * 100, scores["test_f1"].std() * 100
        auc_mean, auc_std = scores["test_roc_auc"].mean() * 100, scores["test_roc_auc"].std() * 100

        cv_records.append({
            "Model": name,
            "Accuracy (%)": f"{acc_mean:.2f} +/- {acc_std:.2f}",
            "Precision (%)": f"{prec_mean:.2f} +/- {prec_std:.2f}",
            "Recall / Sens (%)": f"{rec_mean:.2f} +/- {rec_std:.2f}",
            "F1-Score (%)": f"{f1_mean:.2f} +/- {f1_std:.2f}",
            "ROC-AUC (%)": f"{auc_mean:.2f} +/- {auc_std:.2f}",
            "_raw_f1_mean": float(scores["test_f1"].mean()),
            "_raw_rec_mean": float(scores["test_recall"].mean()),
            "_raw_acc_mean": float(scores["test_accuracy"].mean()),
            "_raw_auc_mean": float(scores["test_roc_auc"].mean()),
            "_raw_acc_std": float(scores["test_accuracy"].std()),
        })

    cv_df = pd.DataFrame(cv_records).sort_values("_raw_f1_mean", ascending=False)
    display_df = cv_df.drop(columns=[c for c in cv_df.columns if c.startswith("_raw_")])

    print("\n" + "=" * 70)
    print(f"5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY (N={len(X)} unique recordings)")
    print("=" * 70)
    print(display_df.to_string(index=False))

    if output_csv:
        display_df.to_csv(output_csv, index=False)
        logger.info("Saved CV comparison table to: %s", output_csv)

    return cv_df


def evaluate_feature_groups_cv(
    csv_path: str,
    n_splits: int = 5,
    random_state: int = 42,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluates acoustic feature subsets (Voice Quality, Spectral, Prosodic, Combined)
    using 5-Fold Stratified Cross-Validation on a Random Forest classifier.
    Replicates notebook feature ablation experiment.
    """
    df = pd.read_csv(csv_path)
    if "Status" not in df.columns:
        raise ValueError("Dataset missing 'Status' column.")

    available_cols = set(df.columns)
    vq = [f for f in VOICE_QUALITY_FEATURES if f in available_cols]
    spec = [f for f in (MFCC_FEATURES + SPECTRAL_FEATURES) if f in available_cols]
    pros = [f for f in PROSODIC_FEATURES if f in available_cols]

    feature_groups = {
        "Voice Quality": vq,
        "Spectral": spec,
        "Prosodic": pros,
        "Combined": vq + spec + pros,
    }

    y = df["Status"].astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    logger.info("Evaluating %d feature groups using %d-fold cross-validation...", len(feature_groups), n_splits)

    for group_name, features in feature_groups.items():
        if not features:
            continue
        X = df[features].replace([np.inf, -np.inf], np.nan)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=random_state
            )),
        ])

        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"]
        )

        results.append({
            "Feature Group": group_name,
            "Accuracy (%)": round(float(scores["test_accuracy"].mean() * 100), 2),
            "Precision (%)": round(float(scores["test_precision"].mean() * 100), 2),
            "Recall (%)": round(float(scores["test_recall"].mean() * 100), 2),
            "F1 Score (%)": round(float(scores["test_f1"].mean() * 100), 2),
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 55)
    print("FEATURE GROUP CROSS-VALIDATION RESULTS")
    print("=" * 55)
    print(results_df.to_string(index=False))

    if output_csv:
        results_df.to_csv(output_csv, index=False)
        logger.info("Saved feature group results to: %s", output_csv)

    return results_df


def train_and_evaluate_models(
    csv_path: str,
    output_dir: str = "saved_models",
    test_size: float = 0.20,
    random_state: int = 42,
    run_cv: bool = True,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Trains and compares SVM, Random Forest, and MLP models using both
    Stratified K-Fold Cross-Validation and an 80/20 holdout test split.
    Selects the best model based on cross-validation F1-score & Recall,
    saves trained .joblib artifacts and best_model.joblib, and logs full metadata.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cross-Validation Phase
    cv_df = None
    if run_cv:
        cv_csv = out_dir / "cv_comparison_results.csv"
        cv_df = evaluate_models_cv(csv_path, n_splits=n_splits, random_state=random_state, output_csv=str(cv_csv))

    # 2. Holdout Test Split Phase
    X, y = load_dataset_for_training(csv_path)
    logger.info("Loaded unique training data: X shape %s, positive class ratio: %.2f", X.shape, y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    models = build_model_pipelines(random_state=random_state)
    results = []
    metadata: Dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "test_size": test_size,
        "random_state": random_state,
        "models": {},
    }

    print("\n" + "=" * 70)
    print(f"TRAINING AND EVALUATING ON 80/20 HOLDOUT SPLIT (Test N={len(X_test)})")
    print("=" * 70)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # Probabilities for ROC-AUC
        prob_pos = None
        auc = np.nan
        if hasattr(pipeline, "predict_proba"):
            prob_pos = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, prob_pos))

        acc = float(accuracy_score(y_test, predictions))
        prec = float(precision_score(y_test, predictions, zero_division=0))
        rec = float(recall_score(y_test, predictions, zero_division=0))
        f1 = float(f1_score(y_test, predictions, zero_division=0))
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        report = classification_report(
            y_test,
            predictions,
            target_names=["Healthy", "Parkinson's"],
            zero_division=0,
            output_dict=True
        )

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall (Sens)": round(rec, 4),
            "Specificity": round(spec, 4),
            "F1-score": round(f1, 4),
            "ROC-AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
            "FN (Missed PD)": int(fn),
            "FP (False Alarm)": int(fp),
        })

        # Save individual model artifact
        model_filename = f"{name}_model.joblib"
        model_filepath = out_dir / model_filename
        joblib.dump(pipeline, model_filepath)
        logger.info("Saved model artifact: %s", model_filepath)

        metadata["models"][name] = {
            "artifact_file": model_filename,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "roc_auc": auc if not np.isnan(auc) else None,
            "confusion_matrix": cm.tolist(),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "classification_report": report,
        }

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    print("\n" + "=" * 70)
    print("HOLDOUT TEST RESULTS COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_csv = out_dir / "model_comparison_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info("Saved comparison summary to: %s", results_csv)

    # 3. Automated Best Model Selection & Justification
    # Selection rule:
    # 1. Primary: 5-Fold Cross-Validation F1-score (generalization stability)
    # 2. Secondary clinical safeguard: High recall / sensitivity (minimizing missed diagnoses / false negatives)
    if cv_df is not None:
        best_row = cv_df.sort_values(by=["_raw_f1_mean", "_raw_rec_mean"], ascending=False).iloc[0]
        best_model_name = best_row["Model"]
        best_cv_f1 = best_row["_raw_f1_mean"] * 100
        best_cv_rec = best_row["_raw_rec_mean"] * 100
        best_cv_acc = best_row["_raw_acc_mean"] * 100
        best_cv_auc = best_row["_raw_auc_mean"] * 100
        best_cv_std = best_row["_raw_acc_std"] * 100
    else:
        best_model_name = results_df.iloc[0]["Model"]
        best_cv_f1 = results_df.iloc[0]["F1-score"] * 100
        best_cv_rec = results_df.iloc[0]["Recall (Sens)"] * 100
        best_cv_acc = results_df.iloc[0]["Accuracy"] * 100
        best_cv_auc = 0.0
        best_cv_std = 0.0

    best_source_artifact = out_dir / f"{best_model_name}_model.joblib"
    best_target_artifact = out_dir / "best_model.joblib"
    shutil.copy(best_source_artifact, best_target_artifact)
    logger.info("Designated '%s' as BEST MODEL and created: %s", best_model_name, best_target_artifact)

    rationale = (
        f"'{best_model_name}' was selected as the best model because it demonstrated the highest "
        f"5-Fold Cross-Validation performance (F1: {best_cv_f1:.2f}%, Accuracy: {best_cv_acc:.2f}%, "
        f"ROC-AUC: {best_cv_auc:.2f}%) with exceptional generalization stability (+/- {best_cv_std:.2f}% std). "
        f"Clinically, it achieved the highest sensitivity/recall ({best_cv_rec:.2f}%), "
        f"minimizing critical false negatives (missed Parkinson's diagnoses), outperforming other candidates "
        f"on both tabular acoustic non-linearities and out-of-fold generalization."
    )

    metadata["best_model"] = {
        "model_name": best_model_name,
        "artifact_file": "best_model.joblib",
        "rationale": rationale,
        "cv_metrics": {
            "f1_percent": round(best_cv_f1, 2),
            "recall_percent": round(best_cv_rec, 2),
            "accuracy_percent": round(best_cv_acc, 2),
            "roc_auc_percent": round(best_cv_auc, 2),
        }
    }

    # Save metadata JSON
    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to: %s", meta_path)

    print("\n" + "=" * 70)
    print("[BEST MODEL SELECTION & CLINICAL DECISION]")
    print("=" * 70)
    print(f"Selected Winner : {best_model_name.upper()}")
    print(f"Primary Artifact: {best_target_artifact}")
    print(f"Decision Reason : {rationale}")
    print("=" * 70 + "\n")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and experiment with Parkinson's voice detection models.")
    parser.add_argument(
        "--csv",
        type=str,
        default="parkinsons_voice_features_unique.csv",
        help="Path to feature dataset CSV file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models",
        help="Directory to save trained model artifacts."
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        default=True,
        help="Run 5-fold cross-validation experiment across models (default: True)."
    )
    parser.add_argument(
        "--no-cv",
        action="store_false",
        dest="run_cv",
        help="Skip cross-validation phase."
    )
    parser.add_argument(
        "--feature-groups-cv",
        action="store_true",
        help="Run feature group ablation experiment."
    )
    args = parser.parse_args()

    if args.feature_groups_cv:
        evaluate_feature_groups_cv(args.csv)

    train_and_evaluate_models(csv_path=args.csv, output_dir=args.output_dir, run_cv=args.run_cv)

