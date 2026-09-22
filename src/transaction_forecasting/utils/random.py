"""Control de semillas para resultados reproducibles."""

import random

import numpy as np

DEFAULT_RANDOM_SEED = 42


def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Fija la semilla de random y NumPy."""
    random.seed(seed)
    np.random.seed(seed)
