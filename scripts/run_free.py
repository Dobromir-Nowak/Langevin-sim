import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity
from langevin_sim.utils.other import load_config
from langevin_sim.physics.langevin import Langevin_sim



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of exact integrals
def f_new(I: np.ndarray, sin_psi:np.ndarray):  # up to order 3
    return (np.pi-2)*I/2 - I**2 * sin_psi / 3 + (3*np.pi - 4)/24 * I**3 * sin_psi**2


f_fn = f_new
I_fn = I_identity

file_name = "free"
config = load_config(file_name=file_name)

# Run simulation 
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn)
r, n = sim.run()
sim.plot_trajectories()
