# dsp/reverb.py
from __future__ import annotations
import numpy as np

# Lightweight Schroeder/Moorer-style reverb:
#  - 4 parallel comb filters (feedback + damping in the loop)
#  - 2 series all-pass diffusers
# Tuned to sound good at 44.1/48 kHz and stay real-time friendly.


class _Comb:
    def __init__(self, delay_samples: int, feedback: float, damp: float):
        self.buf = np.zeros(delay_samples, dtype=np.float32)
        self.idx = 0
        self.feedback = float(feedback)
        self.damp = float(damp)
        self.filter_store = 0.0  # 1-pole lowpass in feedback path

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        b = self.buf
        i = self.idx
        fb = self.feedback
        d = self.damp
        s = self.filter_store

        n = len(x)
        blen = len(b)
        for n0 in range(n):
            y = b[i]
            # simple lowpass in feedback path (damping)
            s = (1 - d) * y + d * s
            b[i] = x[n0] + fb * s
            out[n0] = y
            i += 1
            if i >= blen:
                i = 0

        self.idx = i
        self.filter_store = s
        return out


class _Allpass:
    def __init__(self, delay_samples: int, feedback: float = 0.5):
        self.buf = np.zeros(delay_samples, dtype=np.float32)
        self.idx = 0
        self.feedback = float(feedback)

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        b = self.buf
        i = self.idx
        fb = self.feedback
        n = len(x)
        blen = len(b)

        for n0 in range(n):
            bufout = b[i]
            inp = x[n0]
            y = -inp + bufout
            b[i] = inp + bufout * fb
            out[n0] = y
            i += 1
            if i >= blen:
                i = 0

        self.idx = i
        return out


class Reverb:
    """
    Parameters:
      room_size (0..1): tail length/feedback
      damp      (0..1): high-frequency damping in tail
      pre_delay_ms:     initial pre-delay before reverb builds (feel of space)
    Use:
      y = reverb.process(x, mix=0.2)  # wet/dry mix
    """

    def __init__(
        self,
        sr: int,
        room_size: float = 0.6,
        damp: float = 0.3,
        pre_delay_ms: float = 12.0,
    ):
        self.sr = int(sr)
        self.set_params(room_size, damp, pre_delay_ms)

        # fixed delay taps (scaled for sr) – classic small room feel
        def d(ms):
            return max(1, int(self.sr * ms / 1000.0))

        self.combs = [
            _Comb(d(29.7), self.fb, self.damp),
            _Comb(d(37.1), self.fb, self.damp),
            _Comb(d(41.1), self.fb, self.damp),
            _Comb(d(43.7), self.fb, self.damp),
        ]
        self.allpasses = [
            _Allpass(d(5.0), feedback=0.5),
            _Allpass(d(1.7), feedback=0.5),
        ]
        self.pre_delay_buf = np.zeros(d(self.pre_delay_ms), dtype=np.float32)
        self.pre_idx = 0

    def set_params(self, room_size: float, damp: float, pre_delay_ms: float):
        self.room_size = float(np.clip(room_size, 0.0, 1.0))
        self.damp = float(np.clip(damp, 0.0, 1.0))
        # map room_size to feedback (keep stable)
        self.fb = 0.72 + 0.25 * self.room_size
        self.pre_delay_ms = float(max(0.0, pre_delay_ms))

    def _predelay(self, x: np.ndarray) -> np.ndarray:
        if self.pre_delay_buf.size == 0:
            return x
        out = np.empty_like(x)
        b = self.pre_delay_buf
        i = self.pre_idx
        blen = len(b)
        for n0 in range(len(x)):
            out[n0] = b[i]
            b[i] = x[n0]
            i += 1
            if i >= blen:
                i = 0
        self.pre_idx = i
        return out

    def process(self, x: np.ndarray, mix: float = 0.3) -> np.ndarray:
        # safety & copy-free
        dry = x.astype(np.float32, copy=False)

        # update predelay length if user changed it externally
        want_len = (
            max(1, int(self.sr * self.pre_delay_ms / 1000.0))
            if self.pre_delay_ms > 0
            else 0
        )
        if want_len != self.pre_delay_buf.size:
            self.pre_delay_buf = np.zeros(want_len, dtype=np.float32)
            self.pre_idx = 0

        y = self._predelay(dry)
        # parallel combs
        s = np.zeros_like(y)
        for c in self.combs:
            s += c.process(y)
        s *= 1.0 / len(self.combs)
        # series allpass
        for a in self.allpasses:
            s = a.process(s)

        wet = s
        # Mix with simple headroom and normalization
        wet_peak = np.max(np.abs(wet)) + 1e-9
        dry_peak = np.max(np.abs(dry)) + 1e-9
        wet = wet / wet_peak * dry_peak
        mix = float(np.clip(mix, 0.0, 1.0))
        out = (1 - mix) * dry + mix * wet
        return out
