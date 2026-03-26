import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_config
from Langevin import Langevin_sim
from geometry import Cylinder3D, random_initial_conditions_Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent
plt.style.use(parent_dir / "softmatter.mplstyle")


file_name = "config_Cylinder3D_Gaussian_beam"
config = load_config(file_name=file_name)
sigma_beam = config["sigma_beam"]

# Approximation of f2-f1
def f(x):
    return 0.3*np.sin(x)  # f2-f1 #TODO implement better approximation or original functions
                          # 0.3 comes from f2-f1 approximation

# Light intensity field
def I(r): # r - positions to check field at:
    x, y = r[0,:], r[1,:]
    return np.exp(-(x**2 + y**2)/(2*sigma_beam**2))[None,:] # has to return shape (1, r.shape[1]), r.shape[1] = N
f_fn = f
I_fn = I


# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn)
r_init, n_init = random_initial_conditions_Cylinder3D(config)
sim.reset_and_initialize_state(r_new=r_init, n_new=n_init)
geometry = Cylinder3D(config=config)
sim.geometry = geometry
results = sim.run()
sim.plot_trajectories(n_show=config["N"], aspect_ratio=[2*config["R_cylinder"], 2*config["R_cylinder"], config["zmax"]-config["zmin"]])
# sim.plot_trajectories(n_show=config["N"]) # Version with no scalling