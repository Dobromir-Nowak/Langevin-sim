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


file_name = "side_illumination"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

rm = ResultsManager(config_path=config_path, tag=file_name)


f_fn = F
I_fn = make_const_beam(config=config)


# Geometry
geometry = Cuboid(config=config)


# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([50,50,97],dtype=float)
# n_const = np.array([0,np.sqrt(1)/np.sqrt(10),np.sqrt(9)/np.sqrt(10)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)



sim = Langevin_sim(config, I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry, results_manager=rm)
results = sim.run(save_every=config["save_every"])
n = results["n"]
r = results["r"]


# Plotting

bins = 20
x_max = 100

r_last = r[-1,:]
n_last = n[-1,:]
filt = r_last[0,:]<x_max


r_filt = np.compress(filt, r_last, axis=-1)
n_filt = np.compress(filt, n_last, axis=-1)


pc = PlotCollector()
pc.add(plot_current_ax, r[-1,:], n[-1,:], config, axis_i=0, axis_j=2, bins_xi=bins, bins_xj=bins)
pc.add(plot_current_ax, r_filt, n_filt, config, axis_i=0, axis_j=2, bins_xi=bins, bins_xj=bins, L=np.array([x_max, config["Ly"], config["Lz"]]))
# pc.add(plot_ax, x, y)
# pc.add(plot_current_ax, r[Nth//2,0,:], r[Nth//2,2,:], n[Nth//2,0,:], n[Nth//2,2,:], config)
# pc.add(plot_current_lin_ax, r, n, config, par_vals = angles, axis_r=2, axis_n=0)
rm.save_plot(pc.render(layout="row"), name= f"joint_fig_t={config['Nt']*config['dt']:.0f},N={config["N"]}")
