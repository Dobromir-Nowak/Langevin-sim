from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
from matplotlib.animation import writers

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
        fig.savefig(self.plots / f"{name}.pdf")

    def save_npz(self, name, **arrays):
        np.savez(self.data / f"{name}.npz", **arrays)


    def save_animation(self, ani, name, fps=20, file_format="mp4", dpi=150):
        file_format = file_format.lower().lstrip(".")
        output_path = self.plots / f"{name}.{file_format}"

        if file_format == "mp4":
            if not writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "Matplotlib cannot find ffmpeg. Install ffmpeg and make sure "
                    "it is available on PATH."
                )
            ani.save(
                output_path,
                writer="ffmpeg",
                fps=fps,
                dpi=dpi,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        elif file_format == "gif":
            ani.save(output_path, writer="pillow", fps=fps, dpi=dpi)
        else:
            raise ValueError("file_format must be 'mp4' or 'gif'")

        return output_path


    def save_gif(self, ani, name, save_fps):
        ani.save(self.plots / f"{name}.gif", writer="pillow", fps=save_fps)


