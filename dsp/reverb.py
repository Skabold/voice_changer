import numpy as np
from scipy.signal import fftconvolve

class Reverb:
    def __init__(self, sr: int):
        self.sr = sr
        self.ir = self._make_ir()

    def _make_ir(self, seconds: float = 0.35, decay: float = 3.0, seed: int = 1234):
        n = int(seconds * self.sr)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(n)
        env = np.exp(-decay * np.linspace(0, 1, n))
        ir = noise * env
        ir /= np.max(np.abs(ir)) + 1e-9
        return ir.astype(np.float32)

    def process(self, x: np.ndarray, mix: float = 0.3) -> np.ndarray:
        mix = max(0.0, min(1.0, float(mix)))
        wet = fftconvolve(x, self.ir, mode='full')[: len(x)]
        # normalize wet energy roughly to dry
        wet /= (np.max(np.abs(wet)) + 1e-9)
        wet *= (np.max(np.abs(x)) + 1e-9)
        return (1 - mix) * x + mix * wet