import numpy as np
from scipy.signal import resample

# Very simple block pitch shift by resampling + length match
# (artifact-prone but low-latency and fine for a prototype)

def pitch_shift_block(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    factor = 2 ** (semitones / 12.0)
    n = max(1, int(len(x) / factor))
    y = resample(x, n)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)]