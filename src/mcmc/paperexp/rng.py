"""Random streams keyed by (config_id, stage, rep).

Seeds use the first eight bytes of SHA-256("config_id|stage|rep"), with the
top bit cleared. Separate stages keep calibration, reference and evaluation
draws independent. CPU and CUDA comparisons use pre-drawn noise because
their generators produce different streams.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

_SEP = "|"
_MASK63 = (1 << 63) - 1


def seed_for(config_id: str, stage: str, rep: int) -> int:
    """63-bit seed for one (configuration, estimation stage, repetition) triple."""
    if not isinstance(config_id, str) or not config_id:
        raise ValueError(f"config_id must be a non-empty string, got {config_id!r}")
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"stage must be a non-empty string, got {stage!r}")
    if _SEP in config_id or _SEP in stage:
        raise ValueError(f"'{_SEP}' is reserved: config_id={config_id!r} stage={stage!r}")
    if isinstance(rep, bool) or not isinstance(rep, (int, np.integer)) or rep < 0:
        raise ValueError(f"rep must be a non-negative int, got {rep!r}")
    digest = hashlib.sha256(f"{config_id}{_SEP}{stage}{_SEP}{int(rep)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & _MASK63


def torch_gen(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A torch.Generator on ``device`` seeded with ``seed`` (0 <= seed < 2^63)."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= _MASK63
    ):
        raise ValueError(f"seed must be an int in [0, 2^63), got {seed!r}")
    gen = torch.Generator(device=canonical_device(device))
    gen.manual_seed(int(seed))
    return gen


def canonical_device(device: str | torch.device) -> torch.device:
    """torch.device with an explicit index for CUDA ("cuda" -> "cuda:<current>")."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def np_gen(seed: int) -> np.random.Generator:
    """A numpy PCG64 Generator seeded with ``seed`` (0 <= seed < 2^63)."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= _MASK63
    ):
        raise ValueError(f"seed must be an int in [0, 2^63), got {seed!r}")
    return np.random.Generator(np.random.PCG64(int(seed)))
