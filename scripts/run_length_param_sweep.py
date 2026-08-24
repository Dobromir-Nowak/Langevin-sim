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

# local _ax helper
def plot_cell_fraction_curves_ax(
    ax,
    cell_fraction_curves,
    lower,
    upper,
    t_calibr,
    cell_fraction_calibr,
    calibr_spline,
):
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(cell_fraction_curves))
    )

    for color, (I, t, cell_fraction_t) in zip(
        colors, cell_fraction_curves
    ):
        ax.plot(
            t,
            cell_fraction_t,
            color=color,
            linewidth=1.5,
            label=rf"$I={I:g}$",
        )

    # benchmark
    t_calibr_dense = np.linspace(
        t_calibr.min(),
        t_calibr.max(),
        500,
    )

    ax.plot(
        t_calibr_dense,
        calibr_spline(t_calibr_dense),
        color="black",
        linestyle="-",
        linewidth=2.2,
        zorder=3,
        label="Benchmark spline",
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(r"Cell fraction")
    ax.legend(ncol=2, fontsize="small")


def simulation_time_axis(r, config, t_offset=0.0):
    """
    Time axis for saved simulation frames.

    Parameters
    ----------
    r : ndarray
        Position array with shape (T, dim, N).
    config : dict
        Simulation configuration.
    t_offset : float, optional
        Offset added to the time axis.
    """
    save_every = int(config.get("save_every", 1))
    dt = float(config["dt"])
    return t_offset + np.arange(r.shape[0]) * save_every * dt



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = Path(__file__).stem
config_name = "cuboid_lambda_callibration"
config_path = Path("configs") / f"{config_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F_exact

# Creating the spectrum spline
import pandas as pd
from scipy.interpolate import PchipInterpolator

csv_path = Path("data/mcwhl5_spectrum_digitized_from_plot_5nm.csv")
spec = pd.read_csv(csv_path)

lam_data = spec["wavelength_nm"].to_numpy()
S_data = spec["normalized_intensity"].to_numpy()

lam_normalized = (lam_data - lam_data.min()) / (lam_data.max() - lam_data.min())
S = PchipInterpolator(lam_normalized, S_data, extrapolate=False)


# Result manager
rm = ResultsManager(config_path=config_path, tag=file_name)

# Geometry
geometry = Cuboid(config=config)

# Use identical initial conditions for every intensity
r_init, n_init = geometry.random_initial_conditions()


# Creating the benchmark cell-fraction spline
data_path = Path("data/nb_of_cells_reg_offset.csv")
df = pd.read_csv(data_path)

t_calibr = df["time"].to_numpy()
cell_fraction_calibr = df["cell_fraction_regularized"].to_numpy()

calibr_spline = PchipInterpolator(
    t_calibr,
    cell_fraction_calibr,
    extrapolate=False,
)





# ------------------------------------------------------------
# Run simulation loop
# ------------------------------------------------------------

# Intensity profile params
axis = 1  # y


half_widths = [100, 200] # [100, 200, 400, 600, 800]
ranges = [ (1000-half_width, 1000+half_width) for half_width in half_widths]

# TODO: use a stronger iterative framework to choose I from an
# interval iteratively and save the corresponding calibration data.

axis_length_key = ("Lx", "Ly", "Lz")[axis]

I_vals = np.linspace(0.5, 40.0, 3)

pc = PlotCollector()


for lower, upper in ranges:

    # Keep only the one-dimensional observables required for the final plot,
    # rather than retaining all trajectories from every simulation.
    cell_fraction_curves = []

    vol_frac = (upper - lower) / config[axis_length_key]

    for I in I_vals:
        def base_fn_spline(r: np.ndarray):
            x_i = r[axis, :][None, :]
            x_i_scaled = (x_i - lower) / (upper - lower)
            x_i_scaled = 1.0 - x_i_scaled  # flip to match lambda from 800 to 400
            return I * S(x_i_scaled)

        I_fn = make_gated_intensity(
            base_fn_spline,
            axis=axis,
            lower=lower,
            upper=upper,
        )

        sim = Langevin_sim(
            config,
            I_fn=I_fn,
            f_fn=f_fn,
            r0=r_init,
            n0=n_init,
            geometry=geometry,
        )
        results = sim.run(save_every=config["save_every"])

        r = results["r"]

        mask_t = (lower < r[:, axis, :]) & (r[:, axis, :] < upper)
        cell_count_t = np.sum(mask_t, axis=1)
        cell_fraction_t = cell_count_t / (config["N"] * vol_frac)
        t = simulation_time_axis(r, config)

        cell_fraction_curves.append(
            (float(I), t.copy(), cell_fraction_t.copy())
        )

    # One subplot for this (lower, upper) pair
    pc.add(
        plot_cell_fraction_curves_ax,
        cell_fraction_curves,
        lower,
        upper,
        t_calibr,
        cell_fraction_calibr,
        calibr_spline,
        label=rf"${lower}<y<{upper}$",
    )


# ------------------------------------------------------------
# Combined figure
# ------------------------------------------------------------

fig = pc.render(
    layout="grid",
    ncols=2,
    sharex=True,
    sharey=True,
    show=False,
)

rm.save_plot(
    fig,
    name="Cell_fraction_over_time_ranges"
)

plt.show()