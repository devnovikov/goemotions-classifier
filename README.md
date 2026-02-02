# GoEmotions Multi-Label Emotion Classifier

Multi-label emotion classification system built on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset from Google Research.

## Overview

This project implements an end-to-end machine learning pipeline for classifying emotions in text. Given a piece of text, the model predicts one or more of 28 emotion categories.

### Features

- **Multi-label Classification**: Text can express multiple emotions simultaneously
- **28 Emotion Categories**: Including joy, anger, sadness, fear, surprise, and more
- **Multiple Models**: TF-IDF baseline, RoBERTa, and DeBERTa implementations
- **FastAPI Service**: Production-ready REST API for predictions
- **Docker Support**: Containerized deployment
- **Cross-platform**: Supports CUDA (Windows), MPS (Mac), and CPU

### Emotion Categories

The model classifies text into 28 emotions:

| Category | Category | Category | Category |
|----------|----------|----------|----------|
| admiration | amusement | anger | annoyance |
| approval | caring | confusion | curiosity |
| desire | disappointment | disapproval | disgust |
| embarrassment | excitement | fear | gratitude |
| grief | joy | love | nervousness |
| optimism | pride | realization | relief |
| remorse | sadness | surprise | neutral |

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/goemotions-classifier.git
cd goemotions-classifier

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# For development (includes Jupyter)
uv sync --all-groups
```

## Quick Start

### 1. Explore the Data (EDA)

```bash
uv run jupyter notebook notebooks/notebook.ipynb
```

### 2. Train a Model

```bash
# Train TF-IDF baseline (fast, no GPU required)
uv run python -m src.train.train --model tfidf

# Train RoBERTa (requires GPU or will use CPU)
uv run python -m src.train.train --model roberta

# Train all models
uv run python -m src.train.train --model all
```

### 3. Run the API

```bash
uv run uvicorn src.predict.predict:app --reload
```

### 4. Make Predictions

```bash
# Health check
curl http://localhost:8000/

# Classify text
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'

# Get model info
curl http://localhost:8000/model/info
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Classify emotions in text |
| GET | `/model/info` | Get model metadata |

### Example Request

```json
POST /predict
{
  "text": "I am so happy today!",
  "threshold": 0.35
}
```

### Example Response

```json
{
  "text": "I am so happy today!",
  "labels": ["joy", "optimism"],
  "scores": {
    "joy": 0.85,
    "optimism": 0.72,
    "excitement": 0.45
  },
  "threshold": 0.35,
  "model_type": "tfidf",
  "inference_time_ms": 12.5
}
```

## Docker

### Build

```bash
# First, train a model
uv run python -m src.train.train --model tfidf

# Build the Docker image
docker build -t goemotions-classifier .
```

### Run

```bash
docker run -p 8000:8000 goemotions-classifier
```

### Test

```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

## Project Structure

```
goemotions-classifier/
├── dataset/                 # Downloaded dataset cache
├── models/                  # Saved trained models
│   ├── tfidf_baseline/     # TF-IDF model artifacts
│   ├── roberta/            # RoBERTa model artifacts
│   └── deberta/            # DeBERTa model artifacts
├── notebooks/
│   └── notebook.ipynb      # EDA and experiments
├── screenshots/            # Documentation screenshots
├── src/
│   ├── __init__.py
│   ├── constants.py        # Shared constants
│   ├── data.py             # Dataset loading utilities
│   ├── utils.py            # Helper functions
│   ├── train/
│   │   └── train.py        # Training script
│   └── predict/
│       └── predict.py      # FastAPI service
├── .python-version
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Model Performance

### Baseline (TF-IDF + Logistic Regression)

| Metric | Score |
|--------|-------|
| Macro F1 | ~0.42 |
| Micro F1 | ~0.50 |
| Hamming Loss | ~0.05 |

### Neural Models (RoBERTa/DeBERTa)

| Model | Macro F1 | Micro F1 |
|-------|----------|----------|
| RoBERTa-base | ~0.48 | ~0.55 |
| DeBERTa-v3-large | ~0.52 | ~0.58 |

*Note: Actual results may vary based on training configuration and hardware.*

## Configuration

### Training Options

```bash
python -m src.train.train --help

Options:
  --model         Model type: tfidf, roberta, deberta, all
  --output-dir    Directory to save models (default: models)
  --epochs        Training epochs (neural models)
  --batch-size    Batch size (neural models)
  --learning-rate Learning rate (neural models)
  --seed          Random seed for reproducibility
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FORCE_CPU` | Force CPU usage | Not set |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS memory ratio | Not set |

## Hardware Requirements

### Minimum

- 8GB RAM
- CPU with AVX support

### Recommended

- 16GB RAM
- GPU with 8GB+ VRAM (for neural models)
- CUDA 11.8+ (Windows/Linux) or MPS (Mac M1/M2/M3)

## Dataset

The [GoEmotions](https://arxiv.org/abs/2005.00547) dataset contains ~58,000 Reddit comments labeled with 28 emotion categories.

- **Train**: ~43,000 samples
- **Validation**: ~5,400 samples
- **Test**: ~5,400 samples

The dataset is automatically downloaded from HuggingFace on first use.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) by Google Research
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
