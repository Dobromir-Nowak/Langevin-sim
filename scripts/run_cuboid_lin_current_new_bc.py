import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity, F, const_initial_conditions, const_initial_conditions_split, make_linear_grad_beam
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
from langevin_sim.plotting.plots_ax import PlotCollector, plot_hist_ax, plot_current_ax, plot_density_ax
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cuboid



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "cuboid_lin_new_bc"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F
# I_fn = make_linear_grad_beam(config=config)

sigma_beam = 30
x0 = 50
def I_fn(r):
    x, y = r[0,:], r[1,:]
    return np.exp(-((x-x0)**2)/(2*sigma_beam**2))[None,:]

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
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
nx, ny, nz = n[-1,0,:], n[-1,1,:], n[-1,2,:]


pc = PlotCollector()
pc.add(plot_current_ax, x, z, nx, nz, config, bins_x=20, bins_z=20)
pc.add(plot_density_ax, x, z, config)
pc.add(plot_hist_ax, z, axis_label="z", bins=20)
pc.add(plot_hist_ax, x, axis_label="x", bins=20)
rm.save_plot(pc.render(), name= f"joint_fig_t={config["Nt"]*config["dt"]}")