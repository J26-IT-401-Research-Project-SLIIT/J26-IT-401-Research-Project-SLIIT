import argparse
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_FOLDERS = {
    0: ("SpiralControl", "MeanderControl"),
    1: ("SpiralPatients", "MeanderPatients"),
}


def parse_filename(path):
    match = re.match(r"^(\d+)-(\d+)", path.name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def image_files(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_dataset_root(data_dir):
    required = next(iter(CLASS_FOLDERS.values()))
    candidates = [data_dir, *[path for path in data_dir.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if all((candidate / folder).is_dir() for folder in required):
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset folders below: {data_dir}. "
        "Expected SpiralControl, MeanderControl, SpiralPatients, and MeanderPatients."
    )


def build_pairs(dataset_root):
    rows = []
    for label, (spiral_name, meander_name) in CLASS_FOLDERS.items():
        spiral_folder = dataset_root / spiral_name
        meander_folder = dataset_root / meander_name
        pair_count = 0
        spiral_map = {
            key: path for path in image_files(spiral_folder)
            if (key := parse_filename(path)) is not None
        }
        meander_map = {
            key: path for path in image_files(meander_folder)
            if (key := parse_filename(path)) is not None
        }
        missing = 0
        for (subject_id, image_number), spiral_path in sorted(spiral_map.items()):
            meander_path = meander_map.get((subject_id, image_number + 4))
            if meander_path is None:
                missing += 1
                continue
            prefix = "P" if label else "H"
            pair_count += 1
            rows.append(
                {
                    "spiral": spiral_path,
                    "meander": meander_path,
                    "label": label,
                    "subject": f"{prefix}{subject_id}",
                }
            )
        print(f"{spiral_name}: pairs={pair_count} missing_meander={missing}")
    if not rows:
        raise RuntimeError("No image pairs were found.")
    return rows


def load_image(path, image_size):
    image = tf.keras.utils.load_img(
        path, color_mode="rgb", target_size=(image_size, image_size)
    )
    return np.asarray(image, dtype="float32")


def logit(probability):
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def load_models(model_dir, fold_thresholds):
    models = []
    for fold in sorted(fold_thresholds, key=int):
        model_path = model_dir / f"fold{fold}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        print(f"Loading {model_path.name}")
        models.append((fold, float(fold_thresholds[fold]), tf.keras.models.load_model(model_path)))
    return models


def predict_pairs(rows, models, image_size, batch_size):
    spiral = np.stack([load_image(row["spiral"], image_size) for row in rows])
    meander = np.stack([load_image(row["meander"], image_size) for row in rows])
    probabilities = []
    scores = []
    for fold, threshold, model in models:
        probability = model.predict([spiral, meander], batch_size=batch_size, verbose=0).ravel()
        probabilities.append(probability)
        scores.append(logit(probability) - logit(threshold))
        print(f"Predicted with fold {fold}")
    return np.mean(probabilities, axis=0), np.mean(scores, axis=0)


def run_single(args, config, models):
    if not args.meander:
        raise ValueError("--meander is required when --spiral is provided.")
    rows = [{"spiral": Path(args.spiral), "meander": Path(args.meander), "label": 0, "subject": "single"}]
    probability, score = predict_pairs(rows, models, config["img_size"], args.batch_size)
    prediction = config["label_map"]["1"] if score[0] > 0 else config["label_map"]["0"]
    print(f"Probability: {probability[0]:.6f}")
    print(f"Score:      {score[0]:.6f}")
    print(f"Prediction: {prediction}")


def run_dataset(args, config, models):
    rows = build_pairs(find_dataset_root(args.data_dir))
    if args.limit:
        rows = rows[:args.limit]
    probability, score = predict_pairs(rows, models, config["img_size"], args.batch_size)
    labels = np.array([row["label"] for row in rows])
    predictions = (score > 0).astype(int)

    print(f"\nTest pairs: {len(rows)}")
    if len(np.unique(labels)) == 2:
        print(f"Raw probability AUC: {roc_auc_score(labels, probability):.4f}")
        print(f"Normalized score AUC: {roc_auc_score(labels, score):.4f}")
    print("Confusion matrix [Control, Patient]:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Control", "Patient"],
        zero_division=0,
    ))
    print(f"Balanced accuracy: {balanced_accuracy_score(labels, predictions):.4f}")
    print(f"Macro F1:          {f1_score(labels, predictions, average='macro', zero_division=0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test the HandPD EfficientNet fold ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--spiral", type=Path, help="Spiral image for single-pair prediction.")
    parser.add_argument("--meander", type=Path, help="Meander image for single-pair prediction.")
    parser.add_argument("--limit", type=int, help="Only test the first N dataset pairs.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config_path = args.model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    models = load_models(args.model_dir, config["fold_thresholds"])

    if args.spiral:
        run_single(args, config, models)
    else:
        run_dataset(args, config, models)


if __name__ == "__main__":
    main()