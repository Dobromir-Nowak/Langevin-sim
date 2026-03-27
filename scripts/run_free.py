import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



from langevin_sim.geometry import Cylinder3D

# from langevin_sim.geometry import Cylinder3D

from langevin_sim.utils import load_config
from langevin_sim import langevin
from langevin_sim.geometry import make_geometry_mask

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
f_fn = f
I_fn = I
config = load_config("config_free")

# Run simulation 
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn)
r, n = sim.run()
sim.plot_trajectories()
