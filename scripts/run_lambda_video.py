import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import *
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
from langevin_sim.plotting.plots_ax import *
from langevin_sim.plotting.gifs import make_animation
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import Cuboid

from langevin_sim.plotting.trajectory_animation import make_trajectory_animation

# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "cuboid_lambda_video"
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


# Video params
video_fps = 25
video_frame_stride = 1
video_trail_length = 100   # set to None to show the entire trajectory history
video_n_show = int(config["N"])

# Style override
plt.style.use("default")
#TODO use a stronger iterative framework -- choose "I" from an interval iteratively
#    (+ save the data)

I_vals = np.array([10.]) #np.linspace(0.5, 20., 11) np.array([10.])
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


    # ------------------------------------------------------------
    # Save a 2D y-z trajectory video for this intensity
    # ------------------------------------------------------------

    saved_dt = float(config["dt"]) * int(config["save_every"])

    # with plt.rc_context({"text.usetex": False}):
    animation = make_trajectory_animation(
        r,
        config=config,
        excluded_axis=2,
        n_show=50,
        frame_stride=1,
        trail_length=10,
        fps=25,
        title_func=lambda frame: (
            rf"$I_0={I:.2f},\quad "
            rf"t={frame * config['dt'] * config['save_every']:.1f}\,\mathrm{{s}}$"
        ),
    )
    I_tag = f"{I:.2f}".replace(".", "p")
    video_path = rm.save_animation(
        animation,
        name=f"channel_trajectories_yz_I_{I_tag}",
        fps=video_fps,
        file_format="mp4",
        dpi=150,
    )
    plt.close()
    print(f"Saved video: {video_path}")

