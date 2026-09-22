"""Control de semillas para resultados reproducibles."""

import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """Fija la semilla de random y NumPy."""
    random.seed(seed)
    np.random.seed(seed)
