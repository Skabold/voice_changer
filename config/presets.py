import json
import os


class PresetStore:
    def __init__(self, path: str = "presets.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_all(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_all(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save(self, name: str, preset_dict: dict):
        data = self._load_all()
        data[name] = {
            "pitch": float(preset_dict.get("pitch", 0)),
            "reverb": float(preset_dict.get("reverb", 0)),
        }
        self._save_all(data)

    def load(self, name: str) -> dict:
        return self._load_all().get(name, {})

    def list_names(self):
        return list(self._load_all().keys())
