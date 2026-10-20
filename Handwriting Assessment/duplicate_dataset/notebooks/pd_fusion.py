"""
Handwriting PD score backend
    Spiral model (5 fold models) + Mender model (5 fold models)  ->  one PD score (%)

Place this file in the SAME folder that contains:
    checkpoints/spiral_cv_fold1..5.pt, checkpoints/mender_cv_fold1..5.pt
    results/spiral_cv/{fold_metrics.csv, threshold.json}
    results/mender_cv/{fold_metrics.csv, threshold.json}
(or set the environment variable PD_BASE_DIR, or use --base_dir)

1) Command line:
       python pd_fusion.py --spiral path/to/spiral.png --mender path/to/mender.png

2) API server (for a website / app):
       pip install fastapi uvicorn python-multipart
       uvicorn pd_fusion:app --reload
       -> open http://127.0.0.1:8000/docs  (upload the 2 images and press "Execute")

Both images must be from the SAME person.
Research prototype only - NOT a medical diagnosis tool.
"""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(os.environ.get("PD_BASE_DIR", Path(__file__).resolve().parent))
IMG_SIZE = 160
N_FOLDS = 5
NOTE = "Research prototype only. Not a medical diagnosis."


# --------------------------------------------------- model (same as training)
def conv_block(i, o, drop):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(drop))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 16, 0.05), conv_block(16, 32, 0.10),
                                      conv_block(32, 64, 0.10), conv_block(64, 128, 0.15),
                                      nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 64),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.classifier(self.features(x))


TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])])


def to_image(src):
    """Accepts a file path, raw bytes, or a PIL image -> grayscale PIL image."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    return img.convert("L")


# ------------------------------------------------------------------- engine
class PDEngine:
    def __init__(self, base_dir=BASE_DIR, weight_by="auc"):
        self.base = Path(base_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {k: self._load(k) for k in ("spiral", "mender")}

        self.perf = {}
        for k in ("spiral", "mender"):
            res = self.base / "results" / f"{k}_cv"
            fm = pd.read_csv(res / "fold_metrics.csv")
            self.perf[k] = {"auc": float(fm.auc.mean()), "accuracy": float(fm.accuracy.mean()),
                            "thr": float(json.load(open(res / "threshold.json"))["threshold"])}

        total = sum(self.perf[k][weight_by] for k in self.perf)
        self.w = {k: self.perf[k][weight_by] / total for k in self.perf}
        self.thr = sum(self.w[k] * self.perf[k]["thr"] for k in self.perf)

    def _load(self, kind):
        models = []
        for f in range(1, N_FOLDS + 1):
            path = self.base / "checkpoints" / f"{kind}_cv_fold{f}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            m = CNN().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device)["model"])
            m.eval()
            models.append(m)
        return models

    @torch.no_grad()
    def _score(self, kind, src):
        x = TF(to_image(src)).unsqueeze(0).to(self.device)
        return float(np.mean([torch.sigmoid(m(x)).item() for m in self.models[kind]]))

    def predict(self, spiral, mender):
        s, m = self._score("spiral", spiral), self._score("mender", mender)
        fused = self.w["spiral"] * s + self.w["mender"] * m
        return {
            "spiral_pd_percent": round(s * 100, 1),
            "mender_pd_percent": round(m * 100, 1),
            "handwriting_pd_percent": round(fused * 100, 1),
            "decision_threshold_percent": round(self.thr * 100, 1),
            "label": "Parkinson (PD)" if fused >= self.thr else "Healthy",
            "weights": {k: round(v, 3) for k, v in self.w.items()},
            "model_cv_auc": {k: round(self.perf[k]["auc"], 3) for k in self.perf},
            "note": NOTE,
        }


# ------------------------------------------------------------------ FastAPI
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = PDEngine()
    return _engine


try:
    from fastapi import FastAPI, File, HTTPException, UploadFile

    app = FastAPI(title="Handwriting PD score API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict_endpoint(spiral: UploadFile = File(...), mender: UploadFile = File(...)):
        try:
            return get_engine().predict(await spiral.read(), await mender.read())
        except Exception as e:  # bad image / missing model files
            raise HTTPException(status_code=400, detail=str(e))
except ImportError:  # fastapi not installed -> CLI still works
    app = None


# ---------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spiral", required=True, help="spiral image of the person")
    ap.add_argument("--mender", required=True, help="mender image of the SAME person")
    ap.add_argument("--base_dir", default=str(BASE_DIR))
    ap.add_argument("--weight_by", default="auc", choices=["auc", "accuracy"])
    a = ap.parse_args()

    result = PDEngine(a.base_dir, a.weight_by).predict(a.spiral, a.mender)
    print(json.dumps(result, indent=2))
    print(f"\n==> Handwriting PD score: {result['handwriting_pd_percent']} %  ({result['label']})")
