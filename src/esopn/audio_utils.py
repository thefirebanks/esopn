"""Shared audio helper functions."""

import numpy as np


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono audio using linear interpolation."""
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / float(src_rate)
    new_length = max(1, int(duration * dst_rate))

    old_indices = np.arange(len(audio), dtype=np.float32)
    new_indices = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


def safe_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio if it would otherwise clip."""
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 1.0:
        return (audio / peak).astype(np.float32)
    return audio.astype(np.float32, copy=False)
