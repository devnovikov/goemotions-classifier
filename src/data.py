"""Dataset loading and preprocessing for GoEmotions classifier."""

from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

from .constants import DATASET_CONFIG, DATASET_NAME, EMOTION_LABELS, NUM_LABELS


def load_goemotions(
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """
    Load the GoEmotions dataset from HuggingFace.

    Args:
        split: Specific split to load ('train', 'validation', 'test') or None for all
        cache_dir: Directory to cache the dataset

    Returns:
        Dataset or DatasetDict containing the GoEmotions data
    """
    print(f"Loading GoEmotions dataset ({DATASET_CONFIG})...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        cache_dir=cache_dir,
    )

    print(f"Dataset loaded successfully!")
    if split is None:
        for split_name in dataset:
            print(f"  {split_name}: {len(dataset[split_name])} samples")
    else:
        print(f"  {split}: {len(dataset)} samples")

    return dataset


def get_labels_from_ids(label_ids: list[int]) -> list[str]:
    """
    Convert label IDs to label names.

    Args:
        label_ids: List of label indices

    Returns:
        List of emotion label names
    """
    return [EMOTION_LABELS[i] for i in label_ids]


def create_label_matrix(dataset) -> np.ndarray:
    """
    Create a binary label matrix from the dataset.

    Args:
        dataset: HuggingFace dataset with 'labels' column

    Returns:
        Binary numpy array of shape (n_samples, n_labels)
    """
    n_samples = len(dataset)
    label_matrix = np.zeros((n_samples, NUM_LABELS), dtype=np.float32)

    for i, sample in enumerate(dataset):
        for label_id in sample["labels"]:
            label_matrix[i, label_id] = 1.0

    return label_matrix


def dataset_to_dataframe(dataset) -> pd.DataFrame:
    """
    Convert HuggingFace dataset to pandas DataFrame.

    Args:
        dataset: HuggingFace dataset

    Returns:
        DataFrame with text and label columns
    """
    df = pd.DataFrame(
        {
            "text": dataset["text"],
            "labels": dataset["labels"],
            "label_names": [get_labels_from_ids(ids) for ids in dataset["labels"]],
        }
    )

    # Add individual label columns
    label_matrix = create_label_matrix(dataset)
    for i, label in enumerate(EMOTION_LABELS):
        df[label] = label_matrix[:, i].astype(int)

    return df


def compute_class_weights(dataset) -> np.ndarray:
    """
    Compute class weights for handling class imbalance.

    Uses inverse frequency weighting.

    Args:
        dataset: HuggingFace dataset with 'labels' column

    Returns:
        Array of class weights
    """
    label_matrix = create_label_matrix(dataset)
    class_counts = label_matrix.sum(axis=0)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Inverse frequency weighting
    total_samples = len(dataset)
    weights = total_samples / (NUM_LABELS * class_counts)

    # Normalize to have mean of 1
    weights = weights / weights.mean()

    return weights.astype(np.float32)


def get_dataset_statistics(dataset) -> dict:
    """
    Compute statistics about the dataset.

    Args:
        dataset: HuggingFace dataset

    Returns:
        Dictionary with dataset statistics
    """
    label_matrix = create_label_matrix(dataset)
    texts = dataset["text"]

    stats = {
        "n_samples": len(dataset),
        "n_labels": NUM_LABELS,
        "labels_per_sample": {
            "mean": label_matrix.sum(axis=1).mean(),
            "min": int(label_matrix.sum(axis=1).min()),
            "max": int(label_matrix.sum(axis=1).max()),
        },
        "class_distribution": {
            EMOTION_LABELS[i]: int(label_matrix[:, i].sum())
            for i in range(NUM_LABELS)
        },
        "text_length": {
            "mean": np.mean([len(t) for t in texts]),
            "min": min(len(t) for t in texts),
            "max": max(len(t) for t in texts),
        },
    }

    return stats
