# dsp/pitch.py
import numpy as np
from math import gcd
from scipy.signal import resample_poly


def _ratio_from_factor(factor: float, max_den: int = 1024):
    """Return integers (up, down) ~= factor with small numbers for speed."""
    if factor <= 0:
        return 1, 1
    for scale in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        up = int(round(factor * scale))
        down = scale
        if up == 0:
            up = 1
        g = gcd(up, down)
        up //= g
        down //= g
        if down <= max_den and up <= max_den:
            return up, down
    up = int(round(factor * max_den))
    down = max_den
    g = gcd(up, down)
    return up // g, down // g


class PitchShifter:
    """
    Low-latency pitch shift using resample_poly + short crossfade between blocks.
    Positive semitones -> higher pitch (by resampling with 1/factor).
    """

    def __init__(self, crossfade: int = 128):
        self.semitones = 0.0
        self.crossfade = int(max(0, crossfade))
        self._tail = np.zeros(0, dtype=np.float32)

    def set_semitones(self, semis: float):
        self.semitones = float(semis)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)

        if abs(self.semitones) < 1e-6:
            # smooth exit if a tail existed
            if self._tail.size > 0 and self.crossfade > 0:
                n = min(len(x), self._tail.size, self.crossfade)
                if n > 0:
                    w = np.linspace(1.0, 0.0, n, dtype=np.float32)
                    x[:n] = x[:n] * (1 - w) + self._tail[:n] * w
            self._tail = np.zeros(0, dtype=np.float32)
            return x

        # Correct mapping: pitch factor = 2^(semitones/12); resample by 1/factor
        factor = 2.0 ** (self.semitones / 12.0)
        resample_factor = 1.0 / factor  # <-- FIX
        up, down = _ratio_from_factor(resample_factor)  # <-- FIX

        y = resample_poly(x, up, down, window=("kaiser", 5.0)).astype(np.float32)

        # match length back to block size (stable latency)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        else:
            y = y[: len(x)]

        # short crossfade to hide block edges
        if self._tail.size > 0 and self.crossfade > 0:
            n = min(len(y), self._tail.size, self.crossfade)
            if n > 0:
                w = np.linspace(1.0, 0.0, n, dtype=np.float32)
                y[:n] = y[:n] * (1 - w) + self._tail[:n] * w

        self._tail = (
            y[-self.crossfade :].copy()
            if self.crossfade > 0
            else np.zeros(0, dtype=np.float32)
        )
        return y
