"""Tests for seed module."""

from __future__ import annotations

import random

import numpy as np

from ssda_latent.seed import SeedManager


def test_seed_reproducible() -> None:
    SeedManager.set_all(123)
    a = random.random()
    b = np.random.rand()
    SeedManager.set_all(123)
    assert random.random() == a
    assert np.random.rand() == b
