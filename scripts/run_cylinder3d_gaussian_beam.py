import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import make_I_Gaussian_beam, F
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_density_rho, plot_hist_z, plot_hist_rho
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cylinder3D, random_initial_conditions_Cylinder3D


# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")


file_name = "Cylinder3D_Gaussian_beam"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

sigma_beam = config["sigma_beam"]

rm = ResultsManager(config_path=config_path, tag="cylinder3d_gaussian_beam")


f_fn = F
I_fn = make_I_Gaussian_beam(config)


# Run simulation
r_init, n_init = random_initial_conditions_Cylinder3D(config)
geometry = Cylinder3D(config=config)
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry, results_manager=rm)
results = sim.run(save_every=10000)

# Final results
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)

# Plotting
rm.save_plot(plot_density_rho(rho), name="density_rho")
rm.save_plot(plot_hist_rho(rho), name="hist_rho")
rm.save_plot(plot_hist_z(z), name="hist_z")
sim.plot_trajectories(aspect_ratio=[2*config["R_cylinder"], 2*config["R_cylinder"], config["zmax"]-config["zmin"]])

# sim.plot_trajectories(show_cylinder=True, config=config) # No scalling, cylinder
# sim.plot_trajectories(n_show=config["N"]) # Version with no scalling, all cells