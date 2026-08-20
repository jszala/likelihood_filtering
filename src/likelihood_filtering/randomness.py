from __future__ import annotations

import hashlib

import numpy as np


def _coordinate(value: int | str) -> int:
    if isinstance(value, int):
        return value & 0xFFFFFFFF
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def seeded_rng(master_seed: int, *coordinates: int | str) -> np.random.Generator:
    """Return a generator identified by a seed and semantic coordinates."""
    entropy = [master_seed & 0xFFFFFFFF, *(_coordinate(v) for v in coordinates)]
    return np.random.default_rng(np.random.SeedSequence(entropy))
