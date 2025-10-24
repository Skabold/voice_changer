import tkinter as tk
from tkinter import ttk, messagebox, simpledialog  # simpledialog needed
from audio.engine import AudioEngine, list_devices
from dsp.processor import Processor
from config.presets import PresetStore

APP_TITLE = "Voice Changer Live (Tkinter)"
SR = 44100
BS = 4096


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.resizable(False, False)

        # Core
        self.processor = Processor(sample_rate=SR, block_size=BS)
        self.engine = AudioEngine(
            processor=self.processor, sample_rate=SR, block_size=BS
        )
        self.presets = PresetStore("presets.json")

        # UI state
        self.devices = list_devices()
        self.in_choice = tk.StringVar()
        self.out_choice = tk.StringVar()
        self.enable_processing = tk.BooleanVar(value=True)
        self.monitoring = tk.BooleanVar(value=True)
        self.pitch = tk.DoubleVar(value=0.0)
        self.reverb = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Idle")

        self._build_ui()
        self._populate_devices()
        self._tick()  # periodic sync with DSP

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Input device").grid(row=0, column=0, sticky="w", **pad)
        self.in_combo = ttk.Combobox(
            frm, textvariable=self.in_choice, width=60, state="readonly"
        )
        self.in_combo.grid(row=0, column=1, **pad)

        ttk.Label(frm, text="Output device").grid(row=1, column=0, sticky="w", **pad)
        self.out_combo = ttk.Combobox(
            frm, textvariable=self.out_choice, width=60, state="readonly"
        )
        self.out_combo.grid(row=1, column=1, **pad)

        ttk.Button(frm, text="Refresh devices", command=self._refresh_devices).grid(
            row=2, column=1, sticky="e", **pad
        )

        ttk.Separator(frm, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=10
        )

        # Sliders
        self._add_slider(frm, "Pitch (semitones)", self.pitch, -12, 12, 0.5, row=4)
        self._add_slider(frm, "Reverb amount", self.reverb, 0, 1, 0.05, row=6)

        ttk.Checkbutton(
            frm, text="Enable processing", variable=self.enable_processing
        ).grid(row=7, column=0, sticky="w", **pad)
        ttk.Checkbutton(
            frm, text="Hear myself (monitor)", variable=self.monitoring
        ).grid(row=7, column=1, sticky="w", **pad)

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=2, sticky="ew")
        ttk.Button(btns, text="Start", command=self._start).grid(
            row=0, column=0, padx=5, pady=6
        )
        ttk.Button(btns, text="Stop", command=self._stop).grid(
            row=0, column=1, padx=5, pady=6
        )
        ttk.Button(btns, text="Save preset", command=self._save_preset).grid(
            row=0, column=2, padx=5, pady=6
        )
        ttk.Button(btns, text="Load preset", command=self._load_preset).grid(
            row=0, column=3, padx=5, pady=6
        )

        ttk.Label(frm, textvariable=self.status).grid(
            row=9, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 10)
        )

    def _add_slider(self, parent, label, var, mn, mx, res, row):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        ttk.Label(frm, text=label, width=18).grid(row=0, column=0, sticky="w")
        s = tk.Scale(
            frm,
            from_=mn,
            to=mx,
            resolution=res,
            orient="horizontal",
            variable=var,
            length=360,
        )
        s.grid(row=0, column=1, sticky="ew")
        val_lbl = ttk.Label(frm, text=f"{var.get():.2f}")
        val_lbl.grid(row=0, column=2, padx=6)

        def on_change(*_):
            val_lbl.configure(text=f"{var.get():.2f}")

        var.trace_add("write", on_change)

    # ---------- Devices ----------
    def _device_label(self, d: dict) -> str:
        """Builds a safe label even if hostapi_name is missing."""
        api = d.get("hostapi_name")
        if api is None:
            # fall back to numeric hostapi index or 'Unknown'
            hv = d.get("hostapi")
            api = f"hostapi {hv}" if isinstance(hv, int) else "Unknown"
        idx = d.get("index", -1)
        name = d.get("name", "Device")
        return f"{idx}: [{api}] {name}"

    def _populate_devices(self):
        ins = [d for d in self.devices if d.get("max_input_channels", 0) > 0]
        outs = [d for d in self.devices if d.get("max_output_channels", 0) > 0]
        in_names = [self._device_label(d) for d in ins]
        out_names = [self._device_label(d) for d in outs]
        self.in_combo["values"] = in_names
        self.out_combo["values"] = out_names
        if in_names:
            self.in_choice.set(in_names[0])
        if out_names:
            self.out_choice.set(out_names[0])

    def _refresh_devices(self):
        self.devices = list_devices()
        self._populate_devices()
        self.status.set("Devices refreshed")

    # ---------- Engine control ----------
    def _parse_index(self, s):
        try:
            return int(str(s).split(":", 1)[0])
        except Exception:
            return None

    def _start(self):
        in_idx = self._parse_index(self.in_choice.get())
        out_idx = self._parse_index(self.out_choice.get())
        try:
            self.engine.start(input_device=in_idx, output_device=out_idx)
            self.status.set("Running…")
        except Exception as e:
            messagebox.showerror("Audio error", str(e))
            self.status.set("Error starting")

    def _stop(self):
        self.engine.stop()
        self.status.set("Stopped")

    # ---------- Presets ----------
    def _save_preset(self):
        name = simpledialog.askstring("Save Preset", "Preset name:")
        if not name:
            return
        self.presets.save(
            name,
            {
                "pitch": float(self.pitch.get()),
                "reverb": float(self.reverb.get()),
            },
        )
        self.status.set(f"Saved preset '{name}'")

    def _load_preset(self):
        names = self.presets.list_names()
        if not names:
            messagebox.showinfo("Presets", "No presets saved yet.")
            return
        name = simpledialog.askstring(
            "Load Preset", "Type one of:\n" + "\n".join(names)
        )
        if name and name in names:
            data = self.presets.load(name)
            self.pitch.set(float(data.get("pitch", 0)))
            self.reverb.set(float(data.get("reverb", 0)))
            self.status.set(f"Loaded preset '{name}'")

    # ---------- Periodic sync ----------
    def _tick(self):
        self.processor.set_params(
            enabled=bool(self.enable_processing.get()),
            pitch_semitones=float(self.pitch.get()),
            reverb=float(self.reverb.get()),
        )
        self.engine.set_monitoring(bool(self.monitoring.get()))
        self.after(50, self._tick)


if __name__ == "__main__":
    app = App()
    app.mainloop()
