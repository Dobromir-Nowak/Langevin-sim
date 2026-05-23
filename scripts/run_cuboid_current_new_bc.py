import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import *
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
from langevin_sim.plotting.plots_ax import *
from langevin_sim.plotting.gifs import make_gif
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cuboid



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "cuboid_current_new_bc"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F
# I_fn = make_linear_grad_beam(config=config)

sigma_beam = 10
x0 = 50
# def I_fn(r):
#     x, y = r[0,:], r[1,:]
#     return np.exp(-((x-x0)**2)/(2*sigma_beam**2))[None,:]

I_fn = make_const_beam(config)



rm = ResultsManager(config_path=config_path, tag="cuboid_lin")

# Geometry
geometry = Cuboid(config=config)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)


# Run simulation
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=config["save_every"])

r, n = results["r"], results["n"]

x, y, z = r[:,0,:], r[:,1,:], r[:,2,:]
nx, ny, nz = n[:,0,:], n[:,1,:], n[:,2,:]


# # z histograms as a single plot
# pc = PlotCollector()
# pc.add(plot_hist_lin_ax, r, config, axis=2, t_label=True)
# pc.add(plot_hist_lin_ax, r, config, axis=0, t_label=False)
# rm.save_plot(pc.render(), name= f"test_fig")


# # Orientation decorrelation
# pc = PlotCollector()
# pc.add(plot_n_correlation, n, config, absolute=True)
# pc.add(plot_n_correlation, n, config, absolute=False)
# rm.save_plot(pc.render(), name = "orientation_decorelation")

# # Gifs
# rm.save_gif(make_gif(x, z, nx, nz, plot_func=plot_current_ax, config=config, show=False, fps=10), name="current", save_fps=10)
# rm.save_gif(make_gif(x, plot_func=plot_hist_ax, axis_label="x", show=False, fps=10), name="hist", save_fps=10)

x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
nx, ny, nz = n[-1,0,:], n[-1,1,:], n[-1,2,:]
pc = PlotCollector()
bins_x=20
bins_z=20
pc.add(plot_current_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z)
pc.add(plot_density_ax, x, z, config, bins_x=bins_x, bins_z=bins_z)
pc.add(plot_density_ax, x, y, config, bins_x=bins_x, bins_z=bins_z) # xy
pc.add(plot_hist_ax, z, axis_label=r"z", bins=20)
pc.add(plot_hist_ax, x, axis_label=r"x", bins=20)
pc.add(plot_current_magnitude_and_direction_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z)
pc.add(plot_mean_polarization_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z)
rm.save_plot(pc.render(), name= f"joint_fig_t={config['Nt']*config['dt']}")
