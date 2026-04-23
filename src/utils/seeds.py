"""Seed management utilities.

Rules enforced here:
  - Never call np.random.seed() (global state).
  - Never call torch.manual_seed() outside of train().
  - Sub-seeds for independent uses (train vs. eval) must be derived deterministically.

Usage pattern:
    base_seed = config.training.seed
    train_seed = derive_seed(base_seed, "train")
    eval_seed  = derive_seed(base_seed, "eval")
"""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, purpose: str) -> int:
    """Derive a deterministic sub-seed from a base seed and a purpose string.

    This ensures that train and eval seeds are independent but reproducible.
    The derived seed is in [0, 2^31) to be compatible with both numpy and torch.

    Parameters
    ----------
    base_seed:
        The master seed from the experiment config.
    purpose:
        A short label like "train", "eval", "val_paths".

    Returns
    -------
    int
        A deterministic non-negative integer seed.

    Examples
    --------
    >>> derive_seed(42, "train")  == derive_seed(42, "train")
    True
    >>> derive_seed(42, "train") != derive_seed(42, "eval")
    True
    """
    payload = f"{base_seed}:{purpose}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)
