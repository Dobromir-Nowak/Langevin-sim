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
geometry = Cylinder3D(config=config)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([0,0,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry, results_manager=rm)
results = sim.run(save_every=config["save_every"])



r, n = results["r"], results["n"]

t_idx_list = np.arange(r.shape[0])
# or less crowded:
# t_idx_list = np.array([0, 1, 3, 6])

x, y, z = r[-1, 0, :], r[-1, 1, :], r[-1, 2, :]
rho = np.sqrt(x**2 + y**2)
bins = 10

pc = PlotCollector()

pc.add(
    plot_density_rho_time_ax,
    r,
    config,
    t_idx_list=t_idx_list,
    bins=bins,
    errorbars=True,
    normalize_by_N=True,
)

pc.add(
    plot_density_z_time_ax,
    r,
    config,
    t_idx_list=t_idx_list,
    bins=bins,
    errorbars=True,
    normalize_by_N=True,
    legend=False
)

fig = pc.render(layout="row", show=True)
fig.tight_layout(pad=0.5)

rm.save_plot(
    fig,
    name=f"joint_fig_t={config['Nt'] * config['dt']:.0f},N={config['N']}"
)

plt.close(fig)