import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(path, image_size):
    image = Image.open(path).convert("L")
    image = image.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = 1.0 - array
    return array[..., None][None, ...]


def main():
    parser = argparse.ArgumentParser(
        description="Predict Parkinson's probability from a Spiral/Meander image pair."
    )
    parser.add_argument("--spiral", required=True, help="Path to the Spiral image")
    parser.add_argument("--meander", required=True, help="Path to the Meander image")
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Folder containing fold*.keras and config_*.json files",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_files = sorted(model_dir.glob("config_*.json"))
    if not config_files:
        raise FileNotFoundError(f"No model configuration found in: {model_dir}")

    with config_files[0].open(encoding="utf-8") as file:
        config = json.load(file)

    image_size = tuple(config["img_size"])
    threshold = float(config["threshold"])
    model_files = sorted(model_dir.glob(f"fold*_{image_size[0]}px.keras"))
    if not model_files:
        raise FileNotFoundError(f"No trained fold models found in: {model_dir}")

    spiral = load_image(Path(args.spiral), image_size)
    meander = load_image(Path(args.meander), image_size)

    probabilities = []
    for model_file in model_files:
        model = tf.keras.models.load_model(model_file, compile=False)
        probability = float(model.predict([spiral, meander], verbose=0)[0, 0])
        probabilities.append(probability)

    probability = float(np.mean(probabilities))
    label = "Parkinson's" if probability >= threshold else "Healthy"

    print(f"Spiral image : {args.spiral}")
    print(f"Meander image: {args.meander}")
    print(f"Models used  : {len(model_files)}")
    print(f"PD probability: {probability:.4f}")
    print(f"Threshold     : {threshold:.4f}")
    print(f"Result        : {label}")


if __name__ == "__main__":
    main()
