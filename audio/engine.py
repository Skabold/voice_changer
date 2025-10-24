import sounddevice as sd
import numpy as np
from utils.logger import log


def list_devices():
    infos = sd.query_devices()
    # Attach index explicitly
    return [dict(index=i, **d) for i, d in enumerate(infos)]


class AudioEngine:
    def __init__(self, processor, sample_rate=44100, block_size=1024):
        self.processor = processor
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.stream = None
        self.monitor = True

    def set_monitoring(self, enabled: bool):
        self.monitor = bool(enabled)

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            log(f"Audio status: {status}")
        # Mono only for now; duplicate to output channels if needed
        mono = indata[:, 0].copy() if indata.shape[1] > 0 else np.zeros(frames, dtype=np.float32)
        if self.monitor:
            processed = self.processor.process_block(mono)
        else:
            processed = np.zeros_like(mono)
        # Ensure shape [frames, channels]
        outdata[:, 0] = processed
        # If device has more channels, copy to them
        for ch in range(1, outdata.shape[1]):
            outdata[:, ch] = outdata[:, 0]

    def start(self, input_device=None, output_device=None):
        self.stop()
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype='float32',
            channels=1,
            callback=self._callback,
            device=(input_device, output_device),
        )
        self.stream.start()
        log("Audio engine started")

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
                log("Audio engine stopped")