import argparse
import glob
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "handpd_dataset.zip"
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "model"
IMG_SIZE = 256
BATCH = 16
EPOCHS = 40
N_FOLDS = 5
SEED = 42


def list_imgs(folder):
    paths = set()
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.update(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def find_dir(name):
    matches = [root for root, _, _ in os.walk(DATA_DIR) if os.path.basename(root) == name]
    if not matches:
        raise FileNotFoundError(f"Dataset folder not found: {name}")
    return matches[0]


def prepare_data():
    if not any(os.path.basename(root) == "SpiralPatients" for root, _, _ in os.walk(DATA_DIR)):
        if not ZIP_PATH.exists():
            raise FileNotFoundError(f"Dataset zip not found: {ZIP_PATH}")
        DATA_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_DIR)

    dirs = {name: find_dir(name) for name in (
        "SpiralPatients", "SpiralControl", "MeanderPatients", "MeanderControl"
    )}

    def parse_name(path):
        match = re.match(r"^(\d+)-(\d+)", os.path.basename(path))
        if not match:
            raise ValueError(f"Cannot parse filename: {path}")
        return match.group(1), int(match.group(2))

    def build_pairs(spiral_dir, meander_dir, label):
        spiral = {parse_name(path): path for path in list_imgs(spiral_dir)}
        meander = {parse_name(path): path for path in list_imgs(meander_dir)}
        prefix = "P" if label == 1 else "H"
        rows = []
        missing = 0
        for (subject_id, number), spiral_path in sorted(spiral.items()):
            meander_path = meander.get((subject_id, number + 4))
            if meander_path:
                rows.append((spiral_path, meander_path, label, f"{prefix}{subject_id}"))
            else:
                missing += 1
        print(f"label={label}: pairs={len(rows)}, unmatched={missing}")
        return rows

    rows = build_pairs(dirs["SpiralPatients"], dirs["MeanderPatients"], 1)
    rows += build_pairs(dirs["SpiralControl"], dirs["MeanderControl"], 0)
    frame = pd.DataFrame(rows, columns=["spiral", "meander", "label", "subject"])

    def load_image(path):
        image = tf.keras.utils.load_img(
            path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE)
        )
        array = tf.keras.utils.img_to_array(image) / 255.0
        return (1.0 - array).astype("float32")

    x_spiral = np.stack([load_image(path) for path in frame.spiral])
    x_meander = np.stack([load_image(path) for path in frame.meander])
    labels = frame.label.to_numpy(dtype="float32")
    print(f"Samples: {len(frame)}, subjects: {frame.subject.nunique()}")
    print("Labels:", frame.label.value_counts().to_dict())
    return frame, x_spiral, x_meander, labels


def build_model(steps_per_epoch):
    def branch(name):
        inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1), name=f"{name}_input")
        x = inputs
        for filters in (16, 32, 64, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D()(x)
        features = layers.Concatenate()([
            layers.GlobalAveragePooling2D()(x),
            layers.GlobalMaxPooling2D()(x),
        ])
        return inputs, features

    spiral_input, spiral_features = branch("spiral")
    meander_input, meander_features = branch("meander")
    x = layers.Concatenate()([spiral_features, meander_features])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    network = models.Model([spiral_input, meander_input], output)
    schedule = tf.keras.optimizers.schedules.CosineDecay(5e-4, steps_per_epoch * EPOCHS)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(schedule, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return network


def make_dataset(x_spiral, x_meander, labels, indices, training):
    dataset = tf.data.Dataset.from_tensor_slices(
        ((x_spiral[indices], x_meander[indices]), labels[indices])
    )
    if training:
        dataset = dataset.shuffle(len(indices), seed=SEED)
    return dataset.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def youden_threshold(labels, probabilities):
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    threshold = thresholds[np.argmax(true_positive - false_positive)]
    return float(np.clip(threshold, 1e-6, 1 - 1e-6))


def logit(probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(probabilities / (1.0 - probabilities))


def train():
    tf.keras.utils.set_random_seed(SEED)
    frame, x_spiral, x_meander, labels = prepare_data()
    MODEL_DIR.mkdir(exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probability = np.zeros(len(frame))
    oof_score = np.zeros(len(frame))
    fold_thresholds = []
    fold_aucs = []

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame, frame.label, groups=frame.subject), start=1
    ):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + fold)
        started = time.time()
        count_control = (labels[train_indices] == 0).sum()
        count_patient = (labels[train_indices] == 1).sum()
        class_weight = {
            0: len(train_indices) / (2 * count_control),
            1: len(train_indices) / (2 * count_patient),
        }
        steps = int(np.ceil(len(train_indices) / BATCH))
        network = build_model(steps)
        network.fit(
            make_dataset(x_spiral, x_meander, labels, train_indices, True),
            epochs=EPOCHS,
            class_weight=class_weight,
            verbose=0,
        )
        train_probability = network.predict(
            make_dataset(x_spiral, x_meander, labels, train_indices, False), verbose=0
        ).ravel()
        test_probability = network.predict(
            make_dataset(x_spiral, x_meander, labels, test_indices, False), verbose=0
        ).ravel()
        threshold = youden_threshold(labels[train_indices], train_probability)
        oof_probability[test_indices] = test_probability
        oof_score[test_indices] = logit(test_probability) - logit(threshold)
        fold_thresholds.append(threshold)
        test_auc = roc_auc_score(labels[test_indices], test_probability)
        fold_aucs.append(test_auc)
        network.save(MODEL_DIR / f"fold{fold}.keras")
        print(f"Fold {fold}: test_AUC={test_auc:.3f}, threshold={threshold:.3f}, seconds={time.time() - started:.0f}")

    config = {
        "img_size": IMG_SIZE,
        "fold_thresholds": fold_thresholds,
        "label_map": {"0": "Control", "1": "Patient"},
    }
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    prediction = (oof_score > 0).astype(int)
    print(f"Fold AUC mean +/- std: {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")
    print(f"OOF ROC-AUC: {roc_auc_score(labels, oof_score):.3f}")
    print(confusion_matrix(labels, prediction))
    print(classification_report(labels, prediction, target_names=["Control", "Patient"], zero_division=0))
    print(f"Saved fold models and config to: {MODEL_DIR}")


def test(spiral_path, meander_path):
    with open(MODEL_DIR / "config.json", encoding="utf-8") as file:
        config = json.load(file)

    def prepare(path):
        image = tf.keras.utils.load_img(
            path, color_mode="grayscale", target_size=(config["img_size"], config["img_size"])
        )
        array = tf.keras.utils.img_to_array(image) / 255.0
        return (1.0 - array).astype("float32")[None]

    spiral = prepare(spiral_path)
    meander = prepare(meander_path)
    scores = []
    for fold, threshold in enumerate(config["fold_thresholds"], start=1):
        network = tf.keras.models.load_model(MODEL_DIR / f"fold{fold}.keras")
        probability = float(network.predict([spiral, meander], verbose=0).ravel()[0])
        scores.append(float(logit(probability) - logit(threshold)))
    score = float(np.mean(scores))
    print(json.dumps({
        "score": score,
        "prediction": "Patient" if score > 0 else "Control",
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--spiral", required=True)
    test_parser.add_argument("--meander", required=True)
    arguments = parser.parse_args()
    if arguments.command == "train":
        train()
    else:
        test(arguments.spiral, arguments.meander)
