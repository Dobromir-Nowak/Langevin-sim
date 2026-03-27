import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langevin_sim import Langevin_sim, load_config

# Load plot style
parent_dir = Path(__file__).parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of f2-f1
def f(x):
    return 0.3*np.sin(x)  # f2-f1 #TODO implement better approximation or original functions
                          # 0.3 comes from f2-f1 approximation

# Light intensity field
def I(r): # r - positions to check field at:
    return np.ones((1, r.shape[1]), dtype=float)   # has to return shape (1, r.shape[1]), r.shape[1] = N


def main() -> None:
    config = load_config("config_free")
    sim = Langevin_sim(config, I_fn=I, f_fn=f)
    sim.run()
    sim.plot_trajectories()


if __name__ == "__main__":
    main()
