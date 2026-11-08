"""Regression tests for the explicit attention paths.

The design-pinned kernels are shared by all gate variants, but the
attention-dropout branch is a separate explicit-math path.  These checks
exercise its statistics readout (a shape bug there is invisible in the
kernel path) and the empty-batch forward (a legal data batch whose zero
row count must still produce a legal chunk step).
"""


# [代码审查] 添加随机种子设置以确保实验可复现
import random
def _set_reproducibility(seed=42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

_set_reproducibility()