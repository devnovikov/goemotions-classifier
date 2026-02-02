"""Shared constants for GoEmotions classifier."""

# Random seed for reproducibility
SEED = 42

# GoEmotions emotion labels (28 classes)
EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

NUM_LABELS = len(EMOTION_LABELS)

# Label to index mapping
LABEL2ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

# Model configurations
MODEL_CONFIGS = {
    "tfidf": {
        "name": "TF-IDF + Logistic Regression",
        "max_features": 10000,
        "ngram_range": (1, 2),
    },
    "roberta": {
        "name": "RoBERTa-base",
        "model_name": "roberta-base",
        "max_length": 128,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 3,
    },
    "deberta": {
        "name": "DeBERTa-v3-large",
        "model_name": "microsoft/deberta-v3-large",
        "max_length": 128,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "epochs": 3,
    },
}

# Default classification threshold
DEFAULT_THRESHOLD = 0.35

# Dataset configuration
DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"
