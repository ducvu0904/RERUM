import os
import torch
import numpy as np
import random
import warnings

def seed_everything(seed, seed_cuda=True, strict_cuda=False):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value.
        seed_cuda (bool): Also seed CUDA RNG states when CUDA is available.
        strict_cuda (bool): If True, re-raise CUDA seeding errors.
    """
    # 1. Python & Numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # 2. PyTorch CPU RNG only (safe even when CUDA context is in a bad state)
    torch.random.default_generator.manual_seed(seed)

    # 3. Optional CUDA seeding
    if seed_cuda and torch.cuda.is_available():
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except RuntimeError as exc:
            msg = (
                "CUDA seeding failed (often from a previous CUDA device-side assert). "
                "Restart the notebook kernel to reset CUDA state, or call "
                "seed_everything(seed, seed_cuda=False) to continue on CPU."
            )
            if strict_cuda:
                raise RuntimeError(msg) from exc
            warnings.warn(msg)

    # print(f"Locked random seed: {seed}")
    
def seed_everything_random():
    """
    Tạo random seed, set seed đó, và return seed để bạn biết
    """
    # Tạo random seed
    random_seed = random.randint(0, 999999)
    
    # Set seed using the shared utility above.
    seed_everything(random_seed)
    
    return random_seed