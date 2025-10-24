# dsp/processor.py (only showing the parts that change)
import numpy as np
from .pitch import PitchShifter
from .reverb import Reverb
from .smoothing import Smoother


class Processor:
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sr = sample_rate
        self.bs = block_size
        self.params = {"enabled": True, "pitch_semitones": 0.0, "reverb": 0.0}

        # Pass sample_rate to PitchShifter for proper smoothing
        self.pitch = PitchShifter(
            crossfade=max(96, block_size // 6), sample_rate=self.sr, smooth_ms=35.0
        )
        self.reverb = Reverb(self.sr)

        self.sm_bypass = Smoother(self.sr, time_ms=25.0, initial=1.0)
        self.sm_reverb = Smoother(self.sr, time_ms=30.0, initial=0.0)
        # ⬇️ remove external pitch smoother; we smooth inside PitchShifter
        # self.sm_pitch = Smoother(self.sr, time_ms=40.0, initial=0.0)

        # Optional DC filter state (keep your previous version if you added it)
        self._hp_prev_in = 0.0
        self._hp_prev_out = 0.0
        self._hp_alpha = 0.999

    def set_params(self, **kwargs):
        self.params = {**self.params, **kwargs}

    # ... keep your _dc_remove and _soft_limit helpers if you added them ...

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        n = len(x)

        # (optional) DC removal first if you added it
        # x = self._dc_remove(x)

        # Build smoothed ramps
        bypass_target = 1.0 if self.params.get("enabled", True) else 0.0
        bypass = self.sm_bypass.ramp(n, bypass_target)

        reverb_target = float(np.clip(self.params.get("reverb", 0.0), 0.0, 1.0))
        reverb_mix = self.sm_reverb.ramp(n, reverb_target)

        # ⬇️ Set pitch target directly; PitchShifter smooths internally now
        self.pitch.set_semitones(float(self.params.get("pitch_semitones", 0.0)))

        # --- Process chain ---
        y = x
        if abs(self.pitch.target) > 1e-3:  # use target to avoid an extra call
            y = self.pitch.process_block(y)

        if reverb_target > 1e-4:
            wet = self.reverb.process(y, mix=float(reverb_mix.mean()))
            y = (1.0 - reverb_mix) * y + reverb_mix * wet

        out = (1.0 - bypass) * x + bypass * y

        # (optional) soft limit / final safety normalize if you added it
        # out = self._soft_limit(out, drive=1.1)
        peak = float(np.max(np.abs(out)) + 1e-9)
        if peak > 1.0:
            out = out / peak
        return out
