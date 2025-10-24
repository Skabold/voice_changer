# dsp/pitch.py
import numpy as np
from math import gcd, exp
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
    Internally smooths target semitones over a time constant to avoid zipper/clicks.
    """

    def __init__(
        self, crossfade: int = 128, sample_rate: int = 44100, smooth_ms: float = 35.0
    ):
        self.crossfade = int(max(0, crossfade))
        self.sr = int(sample_rate)
        self.tau = max(1e-3, smooth_ms / 1000.0)  # smoothing time constant
        self.target = 0.0
        self.current = 0.0
        self._tail = np.zeros(0, dtype=np.float32)

    def set_semitones(self, semis: float):
        # just set target; we’ll converge in process_block
        self.target = float(np.clip(semis, -24.0, 24.0))

    def _update_current(self, n_samples: int):
        # exponential convergence over block length
        alpha = 1.0 - exp(-float(n_samples) / (self.tau * self.sr))
        self.current += alpha * (self.target - self.current)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        n = len(x)

        # update smoothed semitone value for this block
        self._update_current(n)
        semis = self.current

        if abs(semis) < 1e-6:
            # smooth exit if a tail existed
            if self._tail.size > 0 and self.crossfade > 0:
                m = min(n, self._tail.size, self.crossfade)
                if m > 0:
                    w = np.linspace(1.0, 0.0, m, dtype=np.float32)
                    x[:m] = x[:m] * (1 - w) + self._tail[:m] * w
            self._tail = np.zeros(0, dtype=np.float32)
            return x

        # pitch factor and correct resample mapping
        factor = 2.0 ** (semis / 12.0)
        resample_factor = 1.0 / factor
        up, down = _ratio_from_factor(resample_factor)

        y = resample_poly(x, up, down, window=("kaiser", 5.0)).astype(np.float32)

        # match length back to block size (stable latency)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        else:
            y = y[:n]

        # short crossfade to hide block edges
        if self._tail.size > 0 and self.crossfade > 0:
            m = min(len(y), self._tail.size, self.crossfade)
            if m > 0:
                w = np.linspace(1.0, 0.0, m, dtype=np.float32)
                y[:m] = y[:m] * (1 - w) + self._tail[:m] * w

        self._tail = (
            y[-self.crossfade :].copy()
            if self.crossfade > 0
            else np.zeros(0, dtype=np.float32)
        )
        return y
