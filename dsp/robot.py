import numpy as np

def robotize(x: np.ndarray, sr: int, depth: float, mod_freq: float = 30.0) -> np.ndarray:
    depth = max(0.0, min(1.0, float(depth)))
    if depth <= 0:
        return x
    t = np.arange(len(x)) / sr
    mod = np.sin(2 * np.pi * mod_freq * t)
    return (1 - depth) * x + depth * (x * mod)