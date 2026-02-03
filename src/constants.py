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
        "max_length": 256,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,  # Same as reference (base), may need 1e-5 for large
        "epochs": 5,
        "warmup_ratio": 0.0,
        "lr_scheduler_type": "linear",
    },
}

# Default classification threshold (0.2 recommended for GoEmotions)
DEFAULT_THRESHOLD = 0.2

# Dataset configuration
DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"

# Emotion colors for visualization (modern palette)
EMOTION_COLORS = {
    # Positive - warm tones
    "joy": "#FFD700",
    "love": "#FF6B9D",
    "excitement": "#FF6B6B",
    "gratitude": "#26DE81",
    "admiration": "#A55EEA",
    "amusement": "#FF9F43",
    "optimism": "#FED330",
    "pride": "#9B59B6",
    "relief": "#7BED9F",
    "approval": "#4ECDC4",
    "caring": "#FF9FF3",
    "desire": "#E84393",
    # Negative - cool tones
    "anger": "#EE5A5A",
    "sadness": "#74B9FF",
    "fear": "#2D3436",
    "disappointment": "#636E72",
    "annoyance": "#FC8181",
    "disgust": "#6C5CE7",
    "grief": "#4A5568",
    "remorse": "#81ECEC",
    "nervousness": "#FDCB6E",
    # Cognitive
    "confusion": "#A29BFE",
    "curiosity": "#00CEC9",
    "realization": "#55EFC4",
    "surprise": "#FD79A8",
    "embarrassment": "#FAB1A0",
    # Neutral
    "neutral": "#95A5A6",
    "disapproval": "#B2BEC3",
}

# Emotion emojis for UI
EMOTION_EMOJIS = {
    "admiration": "🤩",
    "amusement": "😄",
    "anger": "😠",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "😍",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤩",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😢",
    "joy": "😊",
    "love": "❤️",
    "nervousness": "😰",
    "optimism": "🌟",
    "pride": "😌",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😢",
    "surprise": "😲",
    "neutral": "😐",
}
