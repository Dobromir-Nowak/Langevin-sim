import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils import load_config, plot_hist_rho, plot_density_rho, plot_hist_z, const_initial_conditions, I_identity
from langevin_sim.utils import compute_mean_n, compute_autocorr_n
from langevin_sim.langevin import Langevin_sim
from langevin_sim.geometry import Cylinder3D, random_initial_conditions_Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of f2-f1
def f(x):
    return 0.3*np.sin(x)  # f2-f1 #TODO implement better approximation or original functions
                          # 0.3 comes from f2-f1 approximation

f_fn = f
I_fn = I_identity


file_name = "config_Cylinder3D"
config = load_config(file_name=file_name)


# Initial conditions
r_init, n_init = random_initial_conditions_Cylinder3D(config)
# r_init, n_init = None, None # for default initialization

# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init)
geometry = Cylinder3D(config=config)
sim.geometry = geometry
results = sim.run(save_every=10000)

# Final results
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)

_ = compute_mean_n(n)
_ = compute_autocorr_n(n)

# Plotting
# sim.plot_trajectories(aspect_ratio = [1,1,1])
sim.plot_trajectories(aspect_ratio=[2*config["R_cylinder"], 2*config["R_cylinder"], config["zmax"]-config["zmin"]])

plot_density_rho(rho) # plot_hist_rho(rho)
plot_hist_z(z)