# audio/engine.py
import sounddevice as sd
import numpy as np
from utils.logger import log


# --- TOP-LEVEL helper (module function) ---
def list_devices():
    """Return a list of all available audio devices with metadata."""
    try:
        infos = sd.query_devices()
    except Exception as e:
        log(f"Error querying audio devices: {e}")
        return []
    devices = []
    for i, d in enumerate(infos):
        dev = dict(index=i)
        dev.update(d)
        # attach host API name if possible
        try:
            host = sd.query_hostapis(d["hostapi"])
            dev["hostapi_name"] = host.get("name", f"hostapi {d['hostapi']}")
        except Exception:
            dev["hostapi_name"] = "Unknown"
        devices.append(dev)
    return devices


class AudioEngine:
    def __init__(self, processor, sample_rate=44100, block_size=1024):
        self.processor = processor
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.stream = None
        self.monitor = True
        self.input_channels = 1
        self.output_channels = 2

        # fade envelope state
        self._fade_len = int(max(1, 0.03 * self.sample_rate))  # 30ms fade
        self._fade_pos = 0
        self._fading_in = False
        self._fading_out = False

    def set_monitoring(self, enabled: bool):
        self.monitor = bool(enabled)

    def _apply_fade(self, buf: np.ndarray):
        """In-place fade-in/out to avoid clicks when (re)starting/stopping."""
        if not (self._fading_in or self._fading_out):
            return
        n = len(buf)
        if self._fading_in:
            # ramp 0 -> 1
            end = min(self._fade_len - self._fade_pos, n)
            if end > 0:
                ramp = np.linspace(
                    self._fade_pos / self._fade_len,
                    (self._fade_pos + end) / self._fade_len,
                    end,
                    dtype=np.float32,
                )
                buf[:end] *= ramp
                self._fade_pos += end
                if self._fade_pos >= self._fade_len:
                    self._fading_in = False
                    self._fade_pos = 0
        if self._fading_out:
            # ramp 1 -> 0
            end = min(self._fade_len - self._fade_pos, n)
            if end > 0:
                ramp = np.linspace(
                    1.0 - (self._fade_pos / self._fade_len),
                    1.0 - ((self._fade_pos + end) / self._fade_len),
                    end,
                    dtype=np.float32,
                )
                buf[:end] *= ramp
                self._fade_pos += end
                if self._fade_pos >= self._fade_len:
                    self._fading_out = False
                    self._fade_pos = 0

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            log(f"Audio status: {status}")
        mono = (
            indata[:, 0].copy()
            if indata.shape[1] > 0
            else np.zeros(frames, dtype=np.float32)
        )
        processed = (
            self.processor.process_block(mono) if self.monitor else np.zeros_like(mono)
        )

        # apply fade in/out envelope
        self._apply_fade(processed)

        # write to output (duplicate to all channels)
        outdata[:, 0] = processed
        for ch in range(1, outdata.shape[1]):
            outdata[:, ch] = outdata[:, 0]

    def start(self, input_device=None, output_device=None):
        self.stop()
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            channels=(self.input_channels, self.output_channels),  # (in, out)
            callback=self._callback,
            device=(input_device, output_device),  # can be None to use defaults
        )
        self._fading_in = True
        self._fade_pos = 0
        self.stream.start()
        log("Audio engine started")

    def stop(self):
        if self.stream is not None:
            self._fading_out = True
            self._fade_pos = 0
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
                self._fading_out = False
                self._fade_pos = 0
                log("Audio engine stopped")
