import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import F, const_initial_conditions, const_initial_conditions_split, make_const_beam
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cuboid



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "cuboid_const"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F
I_fn = make_const_beam(config=config)

rm = ResultsManager(config_path=config_path, tag=f"cuboid_I={config["I"]}")

# Geometry
geometry = Cuboid(config=config)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation 
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=10000)
# sim.plot_trajectories()

r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]

frac = 0.99
n_z = n[-1,2,:]
n_z_some = n_z[z>frac*config["Lz"]]

# Plotting
rm.save_plot(plot_hist(n_z, axis_label=fr"$\cos\theta$", bins=40, label=fr"Angular distribution of all swimmers, $I={config["I"]}$"), name="hist_n_z")
rm.save_plot(plot_hist(n_z_some, axis_label=fr"$\cos\theta$", bins=40, label=fr"Angular distribution of swimmers with $z>{frac} \cdot L_z$, $I={config["I"]}$"), name=f"hist_n_z_{frac}")