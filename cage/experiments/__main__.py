"""Module entry point: ``python -m cage.experiments``."""

from __future__ import annotations

from . import main
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

if __name__ == "__main__":
    raise SystemExit(main())
