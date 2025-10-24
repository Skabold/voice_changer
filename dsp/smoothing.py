# dsp/smoothing.py
import numpy as np


class Smoother:
    """
    Per-block parameter smoother.
    - Linear ramp from prev -> target over 'time_ms'
    - Generates a per-sample array you can multiply/mix with.
    """

    def __init__(self, sr: int, time_ms: float = 30.0, initial: float = 0.0):
        self.sr = int(sr)
        self.set_time(time_ms)
        self.prev = float(initial)

    def set_time(self, time_ms: float):
        # time in samples for a full ramp
        self.tn = max(1, int(self.sr * (time_ms / 1000.0)))

    def ramp(self, n: int, target: float):
        """Return an array of length n from prev to target, and update prev."""
        p = self.prev
        if n <= 1:
            self.prev = target
            return np.full(n, target, dtype=np.float32)

        # compute how far we can move this block (limit to tn samples)
        # if tn == n, reach the target in exactly one block; if tn > n, partial move.
        step = (target - p) / max(self.tn, 1)
        # generate ramp samples
        out = np.empty(n, dtype=np.float32)
        val = p
        for i in range(n):
            # move one step towards target
            # clamp overshoot
            if (step >= 0 and val + step > target) or (
                step < 0 and val + step < target
            ):
                val = target
            else:
                val += step
            out[i] = val
        # new prev is the last value we produced
        self.prev = float(out[-1])
        return out
