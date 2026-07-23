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

file_name = "cuboid_lambda_callibration"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F

# Creating the spline
import pandas as pd
from scipy.interpolate import PchipInterpolator
csv_path = Path("data/mcwhl5_spectrum_digitized_from_plot_5nm.csv")
spec = pd.read_csv(csv_path)

lam_data = spec["wavelength_nm"].to_numpy()
S_data = spec["normalized_intensity"].to_numpy()

lam_normalized = (lam_data - lam_data.min())/(lam_data.max() - lam_data.min())

S = PchipInterpolator(lam_normalized, S_data, extrapolate=False)


# Result manager
rm = ResultsManager(config_path=config_path, tag=file_name)

# Geometry
geometry = Cuboid(config=config)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()


# Creating the cell fraction spline

data_path = Path("data/nb_of_cells_reg.csv")
df = pd.read_csv(data_path)

t_calibr = df["time"].to_numpy()
cell_fraction_calibr = df["cell_fraction_regularized"].to_numpy()

calibr_spline = PchipInterpolator(t_calibr, cell_fraction_calibr, extrapolate=False)




# ------------------------------------------------------------
# Run simulation loop
# ------------------------------------------------------------


# Intensity profile params

lower = 500
upper = 1500
axis = 1 # y 


#TODO use a stronger iterative framework -- choose "I" from an interval iteratively
#    (+ save the data)

I_vals = np.linspace(0.5, 20., 11)
for I in I_vals:
    def base_fn_spline(r:np.ndarray): 
        x_i = r[axis,:][None,:]
        x_i_scalled = (x_i - lower)/(upper - lower)
        x_i_scalled = 1 - x_i_scalled # flipping to match lambda from 800 to 400
        return I*S(x_i_scalled)   # S - 1d spline

    I_fn = make_gated_intensity(base_fn_spline, axis=axis, lower=lower, upper=upper)
    sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
    results = sim.run(save_every=config["save_every"])

    r, n = results["r"], results["n"]

    x, y, z = r[:,0,:], r[:,1,:], r[:,2,:]
    nx, ny, nz = n[:,0,:], n[:,1,:], n[:,2,:]






# Plotting

def simulation_time_axis(r, config, t_offset=0.0):
    """
    Time axis for saved simulation frames.

    r shape: (T, dim, N)
    """
    save_every = int(config.get("save_every", 1))
    dt = float(config["dt"])
    return t_offset + np.arange(r.shape[0]) * save_every * dt



# ------------------------------------------------------------
# Cell fraction over time
# ------------------------------------------------------------


mask_t = (lower < r[:, 1, :]) & (r[:, 1, :] < upper)
cell_count_t = np.sum(mask_t, axis=1)
vol_frac = (upper-lower)/config["Ly"]
cell_fraction_t = cell_count_t / (config["N"] * vol_frac)
t = simulation_time_axis(r,config)


fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
ax.plot(t, cell_fraction_t)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"Time [s]")
ax.set_ylabel(r"Fraction of cells")
ax.set_title(rf"Fraction of cells such that ${lower}<y<{upper}$ over time")
plt.show()
rm.save_plot(fig, name=f"Cell_fraction_over_time")