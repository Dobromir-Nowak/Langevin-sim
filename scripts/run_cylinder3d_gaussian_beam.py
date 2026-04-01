import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils import load_config, plot_hist_rho, plot_hist_z, make_I_Gaussian_beam
from langevin_sim.langevin import Langevin_sim
from langevin_sim.geometry import Cylinder3D, random_initial_conditions_Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")


file_name = "config_Cylinder3D_Gaussian_beam"
config = load_config(file_name=file_name)
sigma_beam = config["sigma_beam"]

# Approximation of f2-f1
def f(x):
    return 0.3*np.sin(x)  # f2-f1 #TODO implement better approximation or original functions
                          # 0.3 comes from f2-f1 approximation

f_fn = f
I_fn = make_I_Gaussian_beam(config)


# Run simulation
r_init, n_init = random_initial_conditions_Cylinder3D(config)
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init)
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
sim.plot_trajectories(aspect_ratio=[2*config["R_cylinder"], 2*config["R_cylinder"], config["zmax"]-config["zmin"]])
# sim.plot_trajectories(n_show=config["N"]) # Version with no scalling