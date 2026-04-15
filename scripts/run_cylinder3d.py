import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import compute_mean_n, compute_autocorr_n, const_initial_conditions, I_identity
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist_rho, plot_density_rho, plot_hist_z
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cylinder3D, random_initial_conditions_Cylinder3D



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

# Approximation of exact integrals
def f_new(I: np.ndarray, sin_psi:np.ndarray):  # up to order 3
    return (np.pi-2)*I/2 - I**2 * sin_psi / 3 + (3*np.pi - 4)/24 * I**3 * sin_psi**2

f_fn = f_new
I_fn = I_identity

file_name = "Cylinder3D"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

rm = ResultsManager(config_path=config_path)


# Initial conditions
r_init, n_init = random_initial_conditions_Cylinder3D(config)
# r_init, n_init = None, None # for default initialization

# Run simulation
geometry = Cylinder3D(config=config)
sim = Langevin_sim(config,I_fn=I_fn, f_fn=f_fn, r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=10000)

# Final results
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
rho = np.sqrt(x**2+y**2)

_ = compute_mean_n(n)
_ = compute_autocorr_n(n)

# Plotting
# sim.plot_trajectories(aspect_ratio = [1,1,1])
sim.plot_trajectories(aspect_ratio=[2*config["R_cylinder"], 2*config["R_cylinder"], config["zmax"]-config["zmin"]])

plot_density_rho(rho) # plot_hist_rho(rho)
plot_hist_z(z)