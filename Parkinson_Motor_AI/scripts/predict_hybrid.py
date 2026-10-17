from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class HybridMotorCNN(nn.Module):
    def __init__(self, n_features=18):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.handcrafted = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, sensor, manual):
        deep = self.cnn(sensor)
        hand = self.handcrafted(manual)
        return self.fusion(
            torch.cat([deep, hand], dim=1)
        ).squeeze(1)


def predict_participant(participant_id):

    data = ROOT / "data/processed"

    # Only load checkpoints created by your own project.
    ckpt = torch.load(
        ROOT / "models/Hybrid_CNN/PADS_Hybrid_CNN.pt",
        map_location="cpu",
        weights_only=False
    )

    model = HybridMotorCNN(
        len(ckpt["feature_names"])
    )

    model.load_state_dict(
        ckpt["model_state"], strict=True
    )
    model.eval()

    signals = np.load(
        data / "signals.npy", mmap_mode="r"
    )

    features = pd.read_csv(
        data / "features.csv",
        dtype={"participant_id": str}
    )

    metadata = pd.read_csv(
        data / "metadata.csv",
        dtype={"participant_id": str}
    )

    assert len(signals) == len(features) == len(metadata)
    assert features["participant_id"].tolist() == (
        metadata["participant_id"].tolist()
    )

    indices = np.flatnonzero(
        metadata["participant_id"]
        .eq(participant_id).to_numpy()
    )

    if len(indices) == 0:
        raise ValueError(
            f"Participant {participant_id} not found"
        )

    raw = np.asarray(
        signals[indices], dtype=np.float32
    )

    mean = np.asarray(
        ckpt["signal_mean"], dtype=np.float32
    ).reshape(1, 1, 6)

    std = np.asarray(
        ckpt["signal_std"], dtype=np.float32
    ).reshape(1, 1, 6)

    sensor = (
        (raw - mean) / np.maximum(std, 1e-6)
    ).transpose(0, 2, 1).copy()

    manual = features.iloc[indices][
        ckpt["feature_names"]
    ].to_numpy(dtype=np.float32)

    feature_mean = np.asarray(
        ckpt["feature_mean"], dtype=np.float32
    )

    feature_std = np.asarray(
        ckpt["feature_std"], dtype=np.float32
    )

    manual = (
        manual - feature_mean
    ) / np.maximum(feature_std, 1e-6)

    with torch.no_grad():
        logits = model(
            torch.from_numpy(sensor),
            torch.from_numpy(
                manual.astype(np.float32)
            )
        )
        probability = float(
            torch.sigmoid(logits).mean().item()
        )

    result = {
        "participant_id": participant_id,
        "recordings": int(len(indices)),
        "model": "Hybrid CNN",
        "research_prediction_score": probability,
        "predicted_class": (
            "PD" if probability >= 0.5 else "Healthy"
        ),
        "status": "RESEARCH ONLY - NOT CLINICAL"
    }

    return result


if __name__ == "__main__":

    participant_id = (
        sys.argv[1] if len(sys.argv) > 1 else "010"
    )

    result = predict_participant(participant_id)

    print(json.dumps(result, indent=2))

    output_folder = ROOT / "results/inference"
    output_folder.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        output_folder / f"hybrid_{participant_id}.json",
        "w"
    ) as f:
        json.dump(result, f, indent=2)

    print("\nHYBRID CNN INFERENCE COMPLETED!")
