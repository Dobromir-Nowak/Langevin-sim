import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity, F, const_initial_conditions, const_initial_conditions_split, make_linear_grad_beam
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import NewCuboid



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "cuboid_lin_new_bc"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)
alpha = 0.28

f_fn = F
I_fn = make_linear_grad_beam(config=config)



# TODO -- move or remove after testing
def make_D_fn(D_min, D_max, I_min, I_max):
    def D_fn(I):
        D = D_min + (D_max-D_min)*(I-I_min)/(I_max-I_min)
        return D
    return D_fn
I_min = config["offset"]
I_max = config["offset"] + config["Lx"]*config["grad"]
D_fn = make_D_fn(D_min=0.02, D_max=20.,I_min=I_min, I_max=I_max)

rm = ResultsManager(config_path=config_path, tag="cuboid_lin_new_bc")

# Geometry
geometry = NewCuboid(config=config, alpha=alpha)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([0,0,-1],dtype=float) # np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation 
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,D_fn=D_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=1000)
sim.plot_trajectories()

r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
n_z = n[-1,2,:]
n_z_some = n_z[z>0.999*config["Lz"]]

# Plotting
rm.save_plot(plot_hist(x, axis_label="x", bins=20), name="hist_x")
# rm.save_plot(plot_hist(y, axis_label="y", bins=20), name="hist_y")
# rm.save_plot(plot_hist(z, axis_label="z", bins=20), name="hist_z")
# rm.save_plot(plot_hist(n_z, axis_label=fr"$\cos\theta$", bins=40, label="Angular distribution of all swimmers"), name="hist_n_z")
# rm.save_plot(plot_hist(n_z_some, axis_label=fr"$\cos\theta$", bins=40, label=fr"Angular distribution of swimmers with $z>0.999 \cdot L_z$"), name="hist_n_z_some")