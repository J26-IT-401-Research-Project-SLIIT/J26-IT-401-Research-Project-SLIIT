"""
fastapi_example.py
------------------
Sample FastAPI backend application demonstrating how to integrate the
Parkinson's voice classification model into a REST API.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predictor import ParkinsonPredictor

app = FastAPI(
    title="Parkinson's Disease Voice Detection API",
    description="Machine Learning backend for acoustic voice assessment of Parkinson's Disease.",
    version="1.0.0"
)

# Enable CORS for frontend clients (React, Vue, Next.js, mobile apps, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton (loads default model from saved_models/)
predictor = ParkinsonPredictor(model_name="random_forest", models_dir="saved_models")


class FeaturesInput(BaseModel):
    duration_sec: Optional[float] = Field(None, example=2.5)
    Pitch_mean_Hz: Optional[float] = Field(None, example=180.5)
    Pitch_std_Hz: Optional[float] = Field(None, example=15.2)
    Jitter_local: Optional[float] = Field(None, example=0.0065)
    Shimmer_local: Optional[float] = Field(None, example=0.045)
    HNR_dB: Optional[float] = Field(None, example=18.4)
    Voice_Ratio: Optional[float] = Field(None, example=0.88)
    Speech_Rate_proxy: Optional[float] = Field(None, example=3.2)
    Spectral_Centroid: Optional[float] = Field(None, example=1450.0)
    Spectral_Bandwidth: Optional[float] = Field(None, example=1200.0)
    Zero_Crossing_Rate: Optional[float] = Field(None, example=0.045)
    MFCC0: Optional[float] = Field(None, example=-210.0)
    MFCC1: Optional[float] = Field(None, example=110.0)
    MFCC2: Optional[float] = Field(None, example=-15.0)
    MFCC3: Optional[float] = Field(None, example=20.0)
    MFCC4: Optional[float] = Field(None, example=-10.0)
    MFCC5: Optional[float] = Field(None, example=12.0)
    MFCC6: Optional[float] = Field(None, example=-8.0)
    MFCC7: Optional[float] = Field(None, example=5.0)
    MFCC8: Optional[float] = Field(None, example=-6.0)
    MFCC9: Optional[float] = Field(None, example=3.0)
    MFCC10: Optional[float] = Field(None, example=-4.0)
    MFCC11: Optional[float] = Field(None, example=2.0)
    MFCC12: Optional[float] = Field(None, example=-1.5)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Parkinson's Disease Voice Detection API",
        "active_model": predictor.model_name,
        "available_models": predictor.AVAILABLE_MODELS,
    }


@app.get("/models")
def get_available_models():
    """Returns list of supported models."""
    return {
        "active_model": predictor.model_name,
        "supported_models": predictor.AVAILABLE_MODELS
    }


@app.post("/predict/audio")
async def predict_audio_file(
    file: UploadFile = File(..., description="WAV or MP3 audio recording"),
    model: Optional[str] = Query(
        None,
        description="Optional model override ('random_forest', 'svm', 'mlp_neural_network')"
    )
):
    """
    Accepts an audio file upload, extracts all acoustic features,
    and returns a Parkinson's disease risk prediction and confidence score.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported formats: WAV, MP3, OGG, FLAC.")

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = predictor.predict_audio(
            audio_input=audio_bytes,
            filename=file.filename,
            model_override=model
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")


@app.post("/predict/features")
def predict_raw_features(
    features: FeaturesInput,
    model: Optional[str] = Query(None, description="Model override")
):
    """
    Direct prediction endpoint accepting pre-computed acoustic features as JSON.
    """
    try:
        features_dict = features.model_dump(exclude_none=True)
        result = predictor.predict_features(features_dict, model_override=model)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_example:app", host="0.0.0.0", port=8000, reload=True)
