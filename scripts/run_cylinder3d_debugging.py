import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import const_initial_conditions, I_identity
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_density_rho, plot_hist_rho, plot_hist_z
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cylinder3D

# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of exact integrals
def f_new(I: np.ndarray, sin_psi:np.ndarray):  # up to order 3
    return (np.pi-2)*I/2 - I**2 * sin_psi / 3 + (3*np.pi - 4)/24 * I**3 * sin_psi**2


f_fn = f_new
I_fn = I_identity

file_name = "Cylinder3D_debugging"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

results_manager = ResultsManager(config_path=config_path, tag = "test_run")

# Initial conditions
r_const = np.array([0,0,0.9],dtype=float)
n_const = np.array([0,0,1],dtype=float)

# r_const = 0.95*np.array([6,8,1],dtype=float)
# n_const = np.sqrt(2)*np.array([0.6,0.8,0],dtype=float) + np.sqrt(2)*np.array([0,0,1])


r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation
geometry = Cylinder3D(config=config)
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry)
results = sim.run()

# Final results
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)

# Plotting
# plot_hist_rho(rho)
# plot_hist_z(z)
sim.plot_trajectories(aspect_ratio = [1,1,1])