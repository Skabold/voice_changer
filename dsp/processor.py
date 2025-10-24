import numpy as np
from .pitch import pitch_shift_block
from .robot import robotize
from .reverb import Reverb

class Processor:
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sr = sample_rate
        self.bs = block_size
        self.params = {
            'enabled': True,
            'pitch_semitones': 0.0,
            'robot': 0.0,
            'reverb': 0.0,
        }
        self.reverb = Reverb(self.sr)

    def set_params(self, **kwargs):
        # Shallow copy to avoid race in callback
        self.params = {**self.params, **kwargs}

    def process_block(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        y = x.astype(np.float32, copy=False)
        if not p['enabled']:
            return y
        if abs(p['pitch_semitones']) > 1e-3:
            y = pitch_shift_block(y, self.sr, p['pitch_semitones'])
        if p['robot'] > 0:
            y = robotize(y, self.sr, p['robot'])
        if p['reverb'] > 0:
            y = self.reverb.process(y, mix=p['reverb'])
        # keep within [-1,1]
        peak = np.max(np.abs(y)) + 1e-9
        if peak > 1:
            y = y / peak
        return y