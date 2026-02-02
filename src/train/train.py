"""Training script for GoEmotions multi-label emotion classification."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, classification_report
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants import (
    EMOTION_LABELS,
    NUM_LABELS,
    SEED,
    MODEL_CONFIGS,
    DEFAULT_THRESHOLD,
    LABEL2ID,
    ID2LABEL,
)
from src.data import load_goemotions, create_label_matrix, compute_class_weights
from src.utils import setup_environment, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GoEmotions multi-label classifier"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf",
        choices=["tfidf", "roberta", "deberta", "all"],
        help="Model type to train (default: tfidf)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (for neural models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (for neural models)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (for neural models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    return parser.parse_args()


def train_tfidf_baseline(
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    output_dir: str,
) -> dict:
    """
    Train TF-IDF + Logistic Regression baseline model.

    Args:
        train_texts: Training text samples
        train_labels: Training label matrix
        val_texts: Validation text samples
        val_labels: Validation label matrix
        output_dir: Directory to save the model

    Returns:
        Dictionary with training results and metrics
    """
    print("\n" + "=" * 60)
    print("Training TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    config = MODEL_CONFIGS["tfidf"]
    start_time = time.time()

    # Create output directory
    model_dir = Path(output_dir) / "tfidf_baseline"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TF-IDF vectorizer
    print(f"\nInitializing TF-IDF vectorizer...")
    print(f"  max_features: {config['max_features']}")
    print(f"  ngram_range: {config['ngram_range']}")

    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=config["ngram_range"],
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )

    # Fit and transform training data
    print("Fitting TF-IDF on training data...")
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    print(f"  Feature matrix shape: {X_train.shape}")

    # Train OneVsRest classifier with Logistic Regression
    print("\nTraining OneVsRestClassifier with LogisticRegression...")
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        n_jobs=-1,
    )

    classifier.fit(X_train, train_labels)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")

    # Get predictions and probabilities
    print("\nEvaluating on validation set...")
    val_probs = classifier.predict_proba(X_val)

    # Optimize threshold
    best_threshold, best_f1 = optimize_threshold(val_probs, val_labels)
    print(f"Optimal threshold: {best_threshold:.2f}")

    # Get predictions with optimal threshold
    val_preds = (val_probs >= best_threshold).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(val_labels, val_preds, val_probs)
    metrics["threshold"] = best_threshold
    metrics["training_time_seconds"] = training_time

    # Print results
    print_metrics(metrics)

    # Save model
    print(f"\nSaving model to {model_dir}/")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(classifier, model_dir / "classifier.joblib")

    # Save metadata
    metadata = {
        "name": "goemotions-tfidf-baseline",
        "type": "tfidf",
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "num_labels": NUM_LABELS,
        "threshold": best_threshold,
        "config": config,
        "metrics": {
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
            "hamming_loss": metrics["hamming_loss"],
        },
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved successfully!")

    return metrics


def optimize_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> tuple[float, float]:
    """
    Find optimal classification threshold using validation data.

    Args:
        probs: Predicted probabilities
        labels: True labels
        thresholds: List of thresholds to try

    Returns:
        Tuple of (best_threshold, best_f1_score)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.6, 0.05)

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = 0.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """
    Calculate evaluation metrics for multi-label classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
    }

    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_class_f1"] = {
        EMOTION_LABELS[i]: float(per_class_f1[i]) for i in range(NUM_LABELS)
    }

    return metrics


def print_metrics(metrics: dict) -> None:
    """Print formatted metrics."""
    print("\n" + "-" * 40)
    print("EVALUATION METRICS")
    print("-" * 40)
    print(f"Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Samples F1:  {metrics['samples_f1']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")

    if "threshold" in metrics:
        print(f"Threshold:   {metrics['threshold']:.2f}")

    print("\nPer-class F1 scores (sorted):")
    sorted_f1 = sorted(
        metrics["per_class_f1"].items(), key=lambda x: x[1], reverse=True
    )
    for label, score in sorted_f1:
        print(f"  {label:15s}: {score:.4f}")


def train_neural_model(
    model_type: str,
    train_dataset,
    val_dataset,
    output_dir: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Train a neural network model (RoBERTa or DeBERTa).

    Args:
        model_type: 'roberta' or 'deberta'
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Torch device

    Returns:
        Dictionary with training results and metrics
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from torch.utils.data import Dataset

    print("\n" + "=" * 60)
    print(f"Training {model_type.upper()} Neural Model")
    print("=" * 60)

    config = MODEL_CONFIGS[model_type]
    model_name = config["model_name"]

    # Override config with command line args
    epochs = epochs or config["epochs"]
    batch_size = batch_size or config["batch_size"]
    learning_rate = learning_rate or config["learning_rate"]
    max_length = config["max_length"]

    print(f"\nModel: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max length: {max_length}")

    # Setup device
    if device is None:
        device = setup_environment(SEED)

    # Create output directory
    model_dir = Path(output_dir) / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    print("Tokenizing datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    # Create label matrices
    train_labels = create_label_matrix(train_dataset)
    val_labels = create_label_matrix(val_dataset)

    # Custom dataset class
    class EmotionDataset(Dataset):
        def __init__(self, tokenized_data, labels):
            self.tokenized_data = tokenized_data
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {
                "input_ids": torch.tensor(self.tokenized_data[idx]["input_ids"]),
                "attention_mask": torch.tensor(
                    self.tokenized_data[idx]["attention_mask"]
                ),
                "labels": torch.tensor(self.labels[idx], dtype=torch.float),
            }
            return item

    train_ds = EmotionDataset(train_tokenized, train_labels)
    val_ds = EmotionDataset(val_tokenized, val_labels)

    # Load model
    print(f"Loading model from {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Calculate class weights
    class_weights = compute_class_weights(train_dataset)
    class_weights_tensor = torch.tensor(class_weights, device=device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(model_dir / "logs"),
        logging_steps=100,
        seed=SEED,
        fp16=False,
        bf16=torch.cuda.is_available(),
    )

    # Custom trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Compute a class-balanced multi-label loss using BCEWithLogitsLoss.

            This overrides the default Trainer.compute_loss to apply class-balanced
            weighting via torch.nn.BCEWithLogitsLoss, using the precomputed
            class_weights_tensor as the pos_weight argument. This implements
            class-balanced loss for the multi-label emotion classification task.
            """
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= DEFAULT_THRESHOLD).astype(int)
        return {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(labels, preds),
        }

    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()

    # Get predictions for threshold optimization
    predictions = trainer.predict(val_ds)
    val_probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()

    # Optimize threshold
    best_threshold, best_f1 = optimize_threshold(val_probs, val_labels)
    val_preds = (val_probs >= best_threshold).astype(int)

    # Calculate full metrics
    metrics = calculate_metrics(val_labels, val_preds, val_probs)
    metrics["threshold"] = best_threshold
    metrics["training_time_seconds"] = training_time

    print_metrics(metrics)

    # Save model
    print(f"\nSaving model to {model_dir}/")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Save metadata
    metadata = {
        "name": f"goemotions-{model_type}",
        "type": model_type,
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "num_labels": NUM_LABELS,
        "threshold": best_threshold,
        "config": {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
        },
        "metrics": {
            "micro_f1": metrics["micro_f1"],
            "macro_f1": metrics["macro_f1"],
            "hamming_loss": metrics["hamming_loss"],
        },
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model saved successfully!")

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("GoEmotions Multi-Label Emotion Classifier Training")
    print("=" * 60)

    # Setup environment
    set_seed(args.seed)

    # Load dataset
    print("\nLoading GoEmotions dataset...")
    dataset = load_goemotions()

    # Extract texts and labels
    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]
    train_labels = create_label_matrix(dataset["train"])
    val_labels = create_label_matrix(dataset["validation"])

    results = {}

    # Train requested model(s)
    if args.model in ["tfidf", "all"]:
        results["tfidf"] = train_tfidf_baseline(
            train_texts, train_labels, val_texts, val_labels, args.output_dir
        )

    if args.model in ["roberta", "all"]:
        device = setup_environment(args.seed)
        results["roberta"] = train_neural_model(
            "roberta",
            dataset["train"],
            dataset["validation"],
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )

    if args.model in ["deberta", "all"]:
        device = setup_environment(args.seed)
        # Check if CUDA available (DeBERTa-large needs more VRAM)
        if not torch.cuda.is_available():
            print("\nWARNING: DeBERTa-v3-large requires CUDA GPU with 16GB+ VRAM")
            print("Skipping DeBERTa training on this machine.")
        else:
            results["deberta"] = train_neural_model(
                "deberta",
                dataset["train"],
                dataset["validation"],
                args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
            )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.2f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
