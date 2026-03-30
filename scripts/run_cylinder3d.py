import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils import load_config, plot_hist_rho, plot_hist_z
from langevin_sim.langevin import Langevin_sim
from langevin_sim.geometry import Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent.parent
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

# r_init = ...
# n_init = ...
# sim.reset_and_initialize_state(r_new=r_init, n_new=n_init)

geometry = Cylinder3D(config=config)
sim.geometry = geometry
results = sim.run()





# Final results
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)

# Plotting
plot_hist_rho(rho)
plot_hist_z(z)
sim.plot_trajectories()