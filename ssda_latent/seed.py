"""Centralized random seed management."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


class SeedManager:
    @staticmethod
    def set_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
