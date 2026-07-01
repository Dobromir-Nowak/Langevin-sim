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
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([50,50,97],dtype=float)
# n_const = np.array([0,np.sqrt(1)/np.sqrt(10),np.sqrt(9)/np.sqrt(10)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)


# r, n = results["r"], results["n"]
# x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]



# Run multiple simulations 
Nth = 10
angles = np.linspace(0, 0.85, Nth)

sim_ar = []

n = np.zeros((Nth,config["dim"],config["N"]))
r = np.zeros((Nth,config["dim"],config["N"]))

plot_every = 3
indices_selected = np.arange(0, Nth, plot_every) # np.zeros((Nth+1)//2)
angles_selected = angles[indices_selected]
n_selected = np.zeros((len(indices_selected),config["dim"],config["N"]))
r_selected = np.zeros((len(indices_selected),config["dim"],config["N"]))


for idx, angle in enumerate(angles):
    sim = Langevin_sim(config, I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry, results_manager=rm)
    sim.vec_I = np.array([-np.sin(angle),0,-np.cos(angle)])
    results = sim.run(save_every=config["Nt"], is_history=False)
    sim_ar.append(sim)
    n[idx,:,:] = results["n"]
    r[idx,:,:] = results["r"]
    
    if idx in indices_selected:
        n_selected[idx//plot_every,:,:] = results["n"]
        r_selected[idx//plot_every,:,:] = results["r"]


# # Plotting
# sim.plot_trajectories()
# sim.plot_trajectories(n_show=config["N"]) # Version with no scalling, all cells


pc = PlotCollector()
nx = n[:,0,:]
sigma_nx = 1/np.sqrt(config["N"]) * np.std(nx,axis=-1,ddof=1)
pc.add(plot_ax, 180/np.pi*angles, np.mean(nx,axis=-1), y_error=sigma_nx, x_label=r"$\theta\,[\textrm{deg}]$", y_label=r"$\langle \hat{n}_x \rangle$") # y_label=r"$V_s \langle \hat{\mathbf{n}}\rangle$")
pc.add(plot_current_lin_ax, r_selected, n_selected, config, par_vals = angles_selected, axis_r=2, axis_n=0, bins=100, normalize_by_N=False, errorbars=True)
# pc.add(plot_current_lin_ax, r, n, config, par_vals = angles, axis_r=2, axis_n=0) # plot for all angles
rm.save_plot(pc.render(layout="row"), name= f"joint_fig_t={config['Nt']*config['dt']:.0f},N={config["N"]}")
