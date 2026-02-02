"""FastAPI prediction service for GoEmotions classifier."""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import EMOTION_LABELS, NUM_LABELS, DEFAULT_THRESHOLD, ID2LABEL


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to classify",
        examples=["I am so happy today!"],
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Classification threshold (default: model's optimal threshold)",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate that text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""

    text: str
    labels: list[str]
    scores: dict[str, float]
    threshold: float
    model_type: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    service: str
    version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response model for model info endpoint."""

    name: str
    type: str
    version: str
    trained_at: Optional[str]
    num_labels: int
    threshold: float
    metrics: dict


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    message: str
    details: Optional[dict] = None


# ============================================================================
# Model Loading
# ============================================================================

class ModelManager:
    """Manages model loading and inference."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self.model_type = None
        self.tokenizer = None
        self.device = None

    def load_tfidf_model(self, model_dir: Path) -> bool:
        """Load TF-IDF baseline model."""
        try:
            self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
            self.model = joblib.load(model_dir / "classifier.joblib")

            with open(model_dir / "metadata.json") as f:
                self.metadata = json.load(f)

            self.model_type = "tfidf"
            print(f"Loaded TF-IDF model from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading TF-IDF model: {e}")
            return False

    def load_transformer_model(self, model_dir: Path, model_type: str) -> bool:
        """Load transformer model (RoBERTa or DeBERTa)."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

            # Setup device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            self.model.to(self.device)
            self.model.eval()

            with open(model_dir / "metadata.json") as f:
                self.metadata = json.load(f)

            self.model_type = model_type
            print(f"Loaded {model_type} model from {model_dir} on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            return False

    def load_best_available_model(self, models_dir: Path) -> bool:
        """Load the best available model (prefer transformer over TF-IDF)."""
        # Priority: deberta > roberta > tfidf
        for model_type in ["deberta", "roberta", "tfidf_baseline"]:
            model_dir = models_dir / model_type
            if model_dir.exists() and (model_dir / "metadata.json").exists():
                if model_type == "tfidf_baseline":
                    if self.load_tfidf_model(model_dir):
                        return True
                else:
                    if self.load_transformer_model(model_dir, model_type):
                        return True

        print("No trained model found!")
        return False

    def predict(self, text: str, threshold: Optional[float] = None) -> dict:
        """
        Make prediction for a single text.

        Args:
            text: Input text
            threshold: Classification threshold (uses model's default if None)

        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Use model's threshold if not specified
        if threshold is None:
            threshold = self.metadata.get("threshold", DEFAULT_THRESHOLD)

        start_time = time.time()

        if self.model_type == "tfidf":
            probs = self._predict_tfidf(text)
        else:
            probs = self._predict_transformer(text)

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Apply threshold
        predicted_indices = np.where(probs >= threshold)[0]
        predicted_labels = [EMOTION_LABELS[i] for i in predicted_indices]

        # Sort labels by score
        predicted_labels = sorted(
            predicted_labels,
            key=lambda x: probs[EMOTION_LABELS.index(x)],
            reverse=True,
        )

        # Create scores dict (only for labels above a minimum threshold)
        scores = {
            EMOTION_LABELS[i]: float(probs[i])
            for i in range(NUM_LABELS)
            if probs[i] >= 0.1  # Include scores >= 0.1 for context
        }
        # Sort by score
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        return {
            "labels": predicted_labels,
            "scores": scores,
            "threshold": threshold,
            "inference_time_ms": inference_time,
        }

    def _predict_tfidf(self, text: str) -> np.ndarray:
        """Make prediction using TF-IDF model."""
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        return probs

    def _predict_transformer(self, text: str) -> np.ndarray:
        """Make prediction using transformer model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        return probs

    def get_info(self) -> dict:
        """Get model information."""
        if self.metadata is None:
            return {}
        return self.metadata


# ============================================================================
# FastAPI Application
# ============================================================================

# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads model on startup."""
    # Startup
    models_dir = Path(__file__).parent.parent.parent / "models"
    if models_dir.exists():
        model_manager.load_best_available_model(models_dir)
    else:
        print(f"Models directory not found: {models_dir}")
        print("Run training first: python -m src.train.train")

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="GoEmotions Classification API",
    description="Multi-label emotion classification API for the GoEmotions dataset",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns API status and model loading status.
    """
    return HealthResponse(
        status="healthy",
        service="goemotions-classifier",
        version="1.0.0",
        model_loaded=model_manager.model is not None,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Classify emotions in the provided text.

    Returns predicted emotion labels with confidence scores.
    """
    # Check if model is loaded
    if model_manager.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": "No model is currently loaded. Run training first.",
            },
        )

    try:
        result = model_manager.predict(request.text, request.threshold)

        return PredictResponse(
            text=request.text,
            labels=result["labels"],
            scores=result["scores"],
            threshold=result["threshold"],
            model_type=model_manager.model_type,
            inference_time_ms=result["inference_time_ms"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_error",
                "message": str(e),
            },
        )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about the currently loaded model.

    Returns model metadata including version and performance metrics.
    """
    if model_manager.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": "No model is currently loaded. Run training first.",
            },
        )

    info = model_manager.get_info()

    return ModelInfoResponse(
        name=info.get("name", "unknown"),
        type=info.get("type", "unknown"),
        version=info.get("version", "unknown"),
        trained_at=info.get("trained_at"),
        num_labels=info.get("num_labels", NUM_LABELS),
        threshold=info.get("threshold", DEFAULT_THRESHOLD),
        metrics=info.get("metrics", {}),
    )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(
        status_code=400,
        detail={
            "error": "validation_error",
            "message": str(exc),
        },
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
