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


file_name = "Cuboid_PBC"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

rm = ResultsManager(config_path=config_path, tag=file_name)


f_fn = F
I_fn = make_const_beam(config=config)


# Geometry
geometry = Cuboid(config=config)


# Initial conditions
# r_init, n_init = geometry.random_initial_conditions()

r_const = np.array([50,50,97],dtype=float)
n_const = np.array([0,np.sqrt(1)/np.sqrt(10),np.sqrt(9)/np.sqrt(10)],dtype=float)
r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry, results_manager=rm)
results = sim.run(save_every=config["save_every"])

r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)




# Plotting
sim.plot_trajectories(n_show=10)





# # Plotting
# rm.save_plot(plot_density_rho(rho), name="density_rho")
# rm.save_plot(plot_hist_rho(rho), name="hist_rho")
# rm.save_plot(plot_hist_z(z), name="hist_z")
# # sim.plot_trajectories(aspect_ratio=[2*config["R_cylinder"], 2*config["R_cylinder"], config["zmax"]-config["zmin"]])



# sim.plot_trajectories(show_cylinder=True, config=config) # No scalling, cylinder
# sim.plot_trajectories(n_show=config["N"]) # Version with no scalling, all cells



# pc = PlotCollector()
# pc.add(plot_density_rho_ax, rho)
# pc.add(plot_hist_ax, rho, axis_label=r"$\rho$")
# pc.add(plot_hist_ax, z, axis_label=r"$z$")
# rm.save_plot(pc.render(layout="row"), name= f"joint_fig_t={config['Nt']*config['dt']},N={config["N"]}")
