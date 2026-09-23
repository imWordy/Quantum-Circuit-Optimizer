"""Reproducibility utilities for setting random seeds across libraries."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across random, numpy, os, and torch for reproducibility.

    Args:
        seed: Integer seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
