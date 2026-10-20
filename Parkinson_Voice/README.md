# Parkinson's Voice AI - Model & Inference Suite

This module converts the Jupyter Notebook `Parkinson.ipynb` into a clean, modular, production-ready Python codebase designed for training and direct integration into a **FastAPI backend**.

---

## 📁 File Structure

| File | Purpose |
| :--- | :--- |
| [`parkinson.py`](file:///c:/parkinson/Model/parkinson.py) | **All-in-one standalone script**: 1-to-1 conversion of the notebook (Feature extraction -> SHA-256 deduplication -> 3 Model training -> Evaluation). |
| [`feature_extractor.py`](file:///c:/parkinson/Model/feature_extractor.py) | **Audio Feature Engine**: Extracts 23 acoustic features (Praat jitter/shimmer/HNR, Librosa pitch/pyin, MFCCs 0-12, spectral centroid/bandwidth/ZCR). Supports file paths, byte streams, and in-memory buffers. |
| [`dataset.py`](file:///c:/parkinson/Model/dataset.py) | **Dataset Manager**: SHA-256 audio deduplication, label consistency verification, and feature matrix preparation ($X, y$). |
| [`train.py`](file:///c:/parkinson/Model/train.py) | **Model Training & Comparison**: Trains SVM, Random Forest, and MLP Neural Network pipelines, performs Stratified K-Fold CV, evaluates test metrics, and exports `.joblib` model artifacts. |
| [`predictor.py`](file:///c:/parkinson/Model/predictor.py) | **FastAPI Inference Engine**: High-level predictor class (`ParkinsonPredictor`) that takes raw uploaded audio, extracts features, and outputs diagnosis + confidence scores. |
| [`pipeline.py`](file:///c:/parkinson/Model/pipeline.py) | **CLI Orchestrator**: Command-line tool to run individual steps (`extract`, `deduplicate`, `train`, `predict`, `all`). |
| [`fastapi_example.py`](file:///c:/parkinson/Model/fastapi_example.py) | **Starter FastAPI App**: Complete REST API with `/predict/audio` (file upload) and `/predict/features` (JSON body) endpoints. |
| [`requirements.txt`](file:///c:/parkinson/Model/requirements.txt) | **Dependencies**: Required packages for audio processing, scikit-learn, and FastAPI. |

---

## 🛠️ 1. Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 2. Usage Workflows

### Option A: Run Standalone Python Script
To run the full end-to-end pipeline in one go:
```bash
python parkinson.py
```

---

### Option B: Using the CLI Tool (`pipeline.py`)

1. **Feature Extraction** from raw audio directory:
   ```bash
   python pipeline.py extract --audio-dir ../features/archive/Voice_Dataset --output parkinsons_voice_features.csv
   ```

2. **Deduplicate Audio Recordings** using SHA-256 hash matching:
   ```bash
   python pipeline.py deduplicate --csv parkinsons_voice_features.csv --audio-dir ../features/archive/Voice_Dataset --output parkinsons_voice_features_unique.csv
   ```

3. **Train Models** (SVM, Random Forest, MLP):
   ```bash
   python pipeline.py train --csv parkinsons_voice_features_unique.csv --output-dir saved_models/
   ```

4. **Predict a New Audio File**:
   ```bash
   python pipeline.py predict --audio path/to/sample.wav --model random_forest
   ```

---

## ⚡ 3. FastAPI Backend Integration

When building your FastAPI backend, you simply import `ParkinsonPredictor` from `predictor.py`:

```python
from fastapi import FastAPI, UploadFile, File
from predictor import ParkinsonPredictor

app = FastAPI()
predictor = ParkinsonPredictor(model_name="random_forest", models_dir="saved_models")

@app.post("/predict/audio")
async def predict_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = predictor.predict_audio(
        audio_input=audio_bytes,
        filename=file.filename
    )
    return result
```

### Sample API Output:
```json
{
  "prediction": "Parkinson's Disease Detected",
  "status": 1,
  "confidence_score": 0.925,
  "probabilities": {
    "Healthy": 0.075,
    "Parkinson's": 0.925
  },
  "model_used": "random_forest",
  "duration_sec": 2.84,
  "filename": "sample_recording.wav",
  "feature_groups": {
    "voice_quality": {
      "Jitter_local": 0.0071,
      "Shimmer_local": 0.052,
      "HNR_dB": 15.3,
      "Voice_Ratio": 0.84
    },
    "prosodic": {
      "Pitch_mean_Hz": 182.4,
      "Pitch_std_Hz": 18.7,
      "Speech_Rate_proxy": 3.4
    },
    "spectral": {
      "Spectral_Centroid": 1420.5,
      "Spectral_Bandwidth": 1180.2,
      "Zero_Crossing_Rate": 0.048
    },
    "mfcc": {
      "MFCC0": -215.4,
      "MFCC1": 108.2
    }
  }
}
```

To run the sample FastAPI server:
```bash
uvicorn fastapi_example:app --reload --port 8000
```
Then visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.
