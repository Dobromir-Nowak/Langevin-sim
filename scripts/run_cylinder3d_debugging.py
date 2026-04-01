import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils import load_config, plot_hist_rho, plot_hist_z, const_initial_conditions, I_identity
from langevin_sim.langevin import Langevin_sim
from langevin_sim.geometry import Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of f2-f1
def f(x):
    return 0.3*np.sin(x)  # f2-f1 #TODO implement better approximation or original functions
                          # 0.3 comes from f2-f1 approximation

f_fn = f
I_fn = I_identity


file_name = "config_Cylinder3D_debugging"
config = load_config(file_name=file_name)

# Initial conditions
r_const = np.array([0,0,0.9],dtype=float)
n_const = np.array([0,0,1],dtype=float)

# r_const = 0.95*np.array([6,8,1],dtype=float)
# n_const = np.sqrt(2)*np.array([0.6,0.8,0],dtype=float) + np.sqrt(2)*np.array([0,0,1])


r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn)
sim.reset_and_initialize_state(r_new=r_init, n_new=n_init)
geometry = Cylinder3D(config=config)
sim.geometry = geometry
results = sim.run()

# Final results
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)

# Plotting
# plot_hist_rho(rho)
# plot_hist_z(z)
# sim.plot_trajectories(aspect_ratio = [1,1,1])
sim.plot_trajectories(aspect_ratio = [1,1,1], show_cylinder=True, config=config)