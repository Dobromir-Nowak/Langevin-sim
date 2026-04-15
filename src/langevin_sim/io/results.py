from pathlib import Path
from datetime import datetime
import shutil
import numpy as np

class ResultsManager:
    def __init__(self, base_dir="results", config_path=None, tag=None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = timestamp if tag is None else f"{timestamp}_{tag}"

        self.root = Path(base_dir) / name
        self.plots = self.root / "plots"
        self.data = self.root / "data"

        self.root.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir()
        self.data.mkdir()

        if config_path:
            shutil.copy(config_path, self.root / config_path.name)

    def save_plot(self, fig, name):
        fig.savefig(self.plots / f"{name}.png", dpi=600)

    def save_npz(self, name, **arrays):
        np.savez(self.data / f"{name}.npz", **arrays)