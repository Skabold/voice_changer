# dsp/processor.py
import numpy as np
from .pitch import PitchShifter
from .robot import Robotizer
from .reverb import Reverb


class Processor:
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sr = sample_rate
        self.bs = block_size
        self.params = {
            "enabled": True,
            "pitch_semitones": 0.0,
            "robot": 0.0,  # depth 0..1
            "reverb": 0.0,  # mix  0..1
            # optional advanced params later: 'robot_mode': 'vocoder6'|'ring'
        }
        self.pitch = PitchShifter(crossfade=max(64, block_size // 8))
        self.robotizer = Robotizer(self.sr)
        self.reverb = Reverb(self.sr)

    def set_params(self, **kwargs):
        self.params = {**self.params, **kwargs}
        self.pitch.set_semitones(self.params["pitch_semitones"])

    def process_block(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        y = x.astype(np.float32, copy=False)
        if not p["enabled"]:
            return y

        if abs(p["pitch_semitones"]) > 1e-3:
            y = self.pitch.process_block(y)

        if p["robot"] > 0:
            mode = self.params.get("robot_mode", "vocoder6")
            y = self.robotizer.process_block(y, depth=p["robot"], mode=mode)

        if p["reverb"] > 0:
            y = self.reverb.process(y, mix=p["reverb"])

        peak = np.max(np.abs(y)) + 1e-9
        if peak > 1:
            y = y / peak
        return y
