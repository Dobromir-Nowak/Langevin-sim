import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



# TODO adjust imports
from utils import load_config
from Langevin import Langevin_sim
from geometry import Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of f2-f1
def f(x):
    return 0.3*np.sin(x)  # f2-f1 #TODO implement better approximation or original functions
                          # 0.3 comes from f2-f1 approximation

# Light intensity field
def I_identity(r): # r - positions to check field at:
    return np.ones((1, r.shape[1]), dtype=float)   # has to return shape (1, r.shape[1]), r.shape[1] = N
f_fn = f
I_fn = I_identity


file_name = "config_Cylinder3D"
config = load_config(file_name=file_name)



# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn)
geometry = Cylinder3D(config=config)
sim.geometry = geometry
r, n = sim.run()
sim.plot_trajectories()
