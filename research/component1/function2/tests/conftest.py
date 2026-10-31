"""Shared pytest fixtures for Function 2 F2.1A tests.

Ensures the repository root is importable (so ``research.component1...``
absolute imports work the same way whether pytest is invoked from the repo
root or from within this tests/ directory), and provides a few small
synthetic-image fixtures used across test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def flat_gray_image() -> np.ndarray:
    """100x100 BGR image, every pixel (128, 128, 128)."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """64x64 BGR checkerboard, 8px blocks alternating black/white."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i // 8 + j // 8) % 2 == 0:
                img[i:i + 8, j:j + 8] = 255
    return img


@pytest.fixture
def gradient_image() -> np.ndarray:
    """128x128 BGR horizontal luminance gradient, all channels equal (gray)."""
    row = np.linspace(0, 255, 128).astype(np.uint8)
    gray = np.tile(row, (128, 1))
    return np.stack([gray, gray, gray], axis=-1)


@pytest.fixture
def random_noise_image() -> np.ndarray:
    """128x128 BGR uniform random noise, fixed seed for reproducibility."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
