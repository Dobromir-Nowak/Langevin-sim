import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import *
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import *
from langevin_sim.plotting.plots_ax import *
from langevin_sim.plotting.gifs import make_gif
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import *


# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")


file_name = "Cylinder3D_new_bc"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

rm = ResultsManager(config_path=config_path, tag=file_name)


f_fn = F
I_fn = make_I_Gaussian_beam(config)


# Geometry
geometry = Cylinder3D2(config=config)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry, results_manager=rm)
results = sim.run(save_every=config["save_every"])

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





# x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
# nx, ny, nz = n[-1,0,:], n[-1,1,:], n[-1,2,:]
# pc = PlotCollector()
# bins_x=20
# bins_z=20
# pc.add(plot_current_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z)
# pc.add(plot_density_ax, x, z, config, bins_x=bins_x, bins_z=bins_z)
# pc.add(plot_density_ax, x, y, config, bins_x=bins_x, bins_z=bins_z) # xy
# rm.save_plot(pc.render(), name= f"joint_fig_t={config['Nt']*config['dt']}")