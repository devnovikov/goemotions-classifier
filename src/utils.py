"""Utility functions for GoEmotions classifier."""

import os
import random

import numpy as np
import torch

from .constants import SEED


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    return device


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")


def setup_environment(seed: int = SEED) -> torch.device:
    """
    Complete environment setup: set seed and get device.

    Args:
        seed: Random seed value

    Returns:
        torch.device: The selected device
    """
    set_seed(seed)
    return get_device()
