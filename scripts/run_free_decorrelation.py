import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity, make_I_const, F, const_initial_conditions, const_initial_conditions_split, make_linear_grad_beam
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
from langevin_sim.plotting.plots_ax import PlotCollector, plot_hist_ax, plot_hist_lin_ax, plot_current_ax, plot_density_ax, plot_n_correlation
from langevin_sim.plotting.gifs import make_gif
from langevin_sim.io.results import ResultsManager
from langevin_sim.physics.langevin import Langevin_sim



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "free_decorrelation"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F
I_fn = make_I_const(config["magnitude"])

rm = ResultsManager(config_path=config_path, tag=file_name)

# Geometry
geometry = None

# Initial conditions
r_const = np.array([0,0,0],dtype=float)
n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
r_init, _ = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

n_rand = np.random.normal(loc=0,scale=1, size = (config["dim"], config["N"]))
n_init = n_rand/np.linalg.norm(n_rand, axis=0, keepdims=True)

# Run simulation
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=config["save_every"])

r, n = results["r"], results["n"]

x, y, z = r[:,0,:], r[:,1,:], r[:,2,:]
nx, ny, nz = n[:,0,:], n[:,1,:], n[:,2,:]

# Orientation decorrelation
pc = PlotCollector()
# pc.add(plot_n_correlation, n, config, absolute=False, log=True)
pc.add(plot_n_correlation, n, config, absolute=False, log=False)
rm.save_plot(pc.render(layout="column"), name = "orientation_decorelation")
