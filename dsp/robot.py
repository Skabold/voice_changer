# dsp/robot.py
from __future__ import annotations
import numpy as np
from scipy.signal import butter, lfilter


def _one_pole_env(x: np.ndarray, prev: float, alpha: float) -> tuple[np.ndarray, float]:
    """Rectify + 1-pole lowpass envelope follower."""
    x = np.abs(x)
    y = np.empty_like(x)
    s = prev
    a = float(alpha)
    for i in range(len(x)):
        s = a * x[i] + (1 - a) * s
        y[i] = s
    return y, s


def _butter_band(sr: int, f_lo: float, f_hi: float, order: int = 2):
    lo = max(1.0, f_lo) / (0.5 * sr)
    hi = min(0.5 * sr - 100.0, f_hi) / (0.5 * sr)
    hi = max(hi, lo + 1e-5)
    return butter(order, [lo, hi], btype="bandpass")


def _butter_high(sr: int, f_lo: float, order: int = 2):
    wn = max(1.0, f_lo) / (0.5 * sr)
    wn = min(wn, 0.999)
    return butter(order, wn, btype="highpass")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


class _Biquad:
    """IIR wrapper holding zi state so streaming is continuous."""

    def __init__(self, b, a):
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y.astype(np.float32, copy=False)


class _VocoderBand:
    def __init__(self, sr: int, f_lo: float, f_hi: float, env_tau_ms: float = 15.0):
        self.sr = sr
        b, a = _butter_band(sr, f_lo, f_hi, order=2)
        self.bp = _Biquad(b, a)
        # envelope follower time constant
        tau = max(1e-3, env_tau_ms / 1000.0)
        self.env_alpha = 1.0 - np.exp(-1.0 / (tau * sr))
        self.env_prev = 0.0
        # carrier at geometric band center
        self.fc = float(np.sqrt(f_lo * f_hi))
        self.ph = 0.0

    def _carrier(self, n: int) -> np.ndarray:
        inc = 2.0 * np.pi * self.fc / self.sr
        idx = self.ph + inc * np.arange(n, dtype=np.float64)
        self.ph = (self.ph + inc * n) % (2.0 * np.pi)
        return np.sin(idx).astype(np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        band = self.bp.process(x)
        env, self.env_prev = _one_pole_env(band, self.env_prev, self.env_alpha)
        return env.astype(np.float32) * self._carrier(len(x))


class Robotizer:
    """
    Two modes:
      - 'ring': classic ring modulation (metallic)
      - 'vocoder8': 8-band vocoder with unvoiced noise & RMS match (default)
    API:
      y = robot.process_block(x, depth, mode="vocoder8")
    """

    def __init__(self, sr: int):
        self.sr = sr
        # 8 bands ~ 250..8000 Hz (speech region, brighter than 4 kHz)
        edges = np.array(
            [250, 400, 650, 1000, 1600, 2500, 4000, 6000, 8000], dtype=float
        )
        self.bands = [_VocoderBand(sr, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
        # highpass noise generator for unvoiced consonants
        b, a = _butter_high(sr, 4000.0, order=1)
        self.hp_noise = _Biquad(b, a)
        self._rng = np.random.default_rng(12345)
        self._hf_env_prev = 0.0
        # envelope for HF energy to drive unvoiced noise
        tau = 0.008  # faster envelope for fricatives
        self._hf_alpha = 1.0 - np.exp(-1.0 / (tau * sr))

    def ring_mod(self, x: np.ndarray, freq: float = 30.0) -> np.ndarray:
        t = np.arange(len(x), dtype=np.float32) / self.sr
        mod = np.sin(2 * np.pi * freq * t).astype(np.float32)
        y = x * mod
        # simple DC-removal (high-pass one-pole)
        hp = np.empty_like(y)
        alpha = 0.995
        prev = 0.0
        for i in range(len(y)):
            prev = alpha * (prev + y[i] - (y[i - 1] if i > 0 else 0.0))
            hp[i] = prev
        return hp

    def _unvoiced_noise(self, n: int, drive: float) -> np.ndarray:
        # white noise -> highpass -> scale by drive
        noise = self._rng.standard_normal(n).astype(np.float32) * 0.5
        noise_hp = self.hp_noise.process(noise)
        return noise_hp * float(np.clip(drive, 0.0, 1.0))

    def vocoder8(self, x: np.ndarray) -> np.ndarray:
        # Sum bands
        acc = np.zeros_like(x, dtype=np.float32)
        for b in self.bands:
            acc += b.process(x)
        acc *= 1.0 / len(self.bands)

        # High-frequency envelope for unvoiced mix
        # Use the top band as proxy for fricatives
        top_band = self.bands[-1].bp.process(x)
        hf_env, self._hf_env_prev = _one_pole_env(
            top_band, self._hf_env_prev, self._hf_alpha
        )
        # normalize envelope roughly to 0..1
        if np.max(hf_env) > 1e-6:
            hf_drive = (hf_env / (np.max(hf_env) + 1e-9)).astype(np.float32)
        else:
            hf_drive = np.zeros_like(acc)

        # Add a little unvoiced noise where fricatives happen
        noise = self._unvoiced_noise(len(x), drive=np.mean(hf_drive))
        out = acc + 0.3 * noise  # keep subtle
        return out

    def process_block(
        self, x: np.ndarray, depth: float, mode: str = "vocoder8"
    ) -> np.ndarray:
        depth = float(np.clip(depth, 0.0, 1.0))
        dry = x.astype(np.float32, copy=False)

        if depth <= 0.0:
            return dry

        wet = self.ring_mod(dry) if mode == "ring" else self.vocoder8(dry)

        # --- Loudness compensation (match wet RMS to dry RMS) ---
        rms_w = _rms(wet)
        rms_d = _rms(dry)
        if rms_w > 0:
            wet = wet * (rms_d / rms_w)

        # --- Minimum dry floor to keep intelligibility, even at depth=1 ---
        min_dry = 0.15  # tweak: 0.10..0.25
        mix_wet = depth
        mix_dry = max(1.0 - depth, min_dry)

        # renormalize so total <= 1.0
        total = mix_wet + mix_dry
        mix_wet /= total
        mix_dry /= total

        out = mix_dry * dry + mix_wet * wet

        # safety limiter
        peak = np.max(np.abs(out)) + 1e-9
        if peak > 1.0:
            out = out / peak
        return out
