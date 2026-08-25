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

import pandas as pd
from scipy.interpolate import PchipInterpolator


# ------------------------------------------------------------
# Local helpers
# ------------------------------------------------------------

def plot_cell_fraction_curves_ax(
    ax,
    cell_fraction_curves,
    t_calibr,
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


def plot_mse_grid_ax(
    ax,
    I_vals,
    D_r_vals,
    mse_grid,
):
    mesh = ax.pcolormesh(
        I_vals,
        D_r_vals,
        mse_grid,
        shading="nearest",
        cmap="viridis",
    )

    fig = ax.get_figure()
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("MSE")

    ax.set_xlabel(r"$I$")
    ax.set_ylabel(r"$D_r$")
    ax.set_yscale("log")

    i_D_r, i_I = np.unravel_index(
        np.argmin(mse_grid),
        mse_grid.shape,
    )

    ax.plot(
        I_vals[i_I],
        D_r_vals[i_D_r],
        marker="x",
        markersize=10,
        markeredgewidth=2,
        color="red",
    )


def simulation_time_axis(r, config, t_offset=0.0):
    save_every = int(config.get("save_every", 1))
    dt = float(config["dt"])
    return t_offset + np.arange(r.shape[0]) * save_every * dt


def calibration_mse(
    t,
    cell_fraction_t,
    calibr_spline,
):
    # Evaluate the calibration spline at the simulation time points.
    mask = (
        (t >= calibr_spline.x[0])
        & (t <= calibr_spline.x[-1])
    )

    return np.mean(
        (
            cell_fraction_t[mask]
            - calibr_spline(t[mask])
        ) ** 2
    )


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

parent_dir = Path.cwd()
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "run_lambda_optim"
config_name = "cuboid_lambda_callibration"
config_path = Path("configs") / f"{config_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F_exact


# Spectrum spline
csv_path = Path("data/mcwhl5_spectrum_digitized_from_plot_5nm.csv")
spec = pd.read_csv(csv_path)

lam_data = spec["wavelength_nm"].to_numpy()
S_data = spec["normalized_intensity"].to_numpy()

lam_normalized = (
    (lam_data - lam_data.min())
    / (lam_data.max() - lam_data.min())
)

S = PchipInterpolator(
    lam_normalized,
    S_data,
    extrapolate=False,
)


# Results
rm = ResultsManager(
    config_path=config_path,
    tag=file_name,
)


# Geometry and common initial conditions
geometry = Cuboid(config=config)
r_init, n_init = geometry.random_initial_conditions()


# Calibration spline
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
# Single simulation
# ------------------------------------------------------------

axis = 1  # y


def sim_run(
    config,
    f_fn,
    r0,
    n0,
    geometry,
    lower,
    upper,
    I,
    D_r,
):
    config_run = config.copy()
    config_run["D_r"] = D_r

    def base_fn_spline(r: np.ndarray):
        x_i = r[axis, :][None, :]
        x_i_scaled = (x_i - lower) / (upper - lower)
        x_i_scaled = 1.0 - x_i_scaled
        return I * S(x_i_scaled)

    I_fn = make_gated_intensity(
        base_fn_spline,
        axis=axis,
        lower=lower,
        upper=upper,
    )

    sim = Langevin_sim(
        config_run,
        I_fn=I_fn,
        f_fn=f_fn,
        r0=r0,
        n0=n0,
        geometry=geometry,
    )

    results = sim.run(
        save_every=config_run["save_every"]
    )

    r = results["r"]

    mask_t = (
        (lower < r[:, axis, :])
        & (r[:, axis, :] < upper)
    )

    cell_count_t = np.sum(mask_t, axis=1)

    axis_length_key = ("Lx", "Ly", "Lz")[axis]
    vol_frac = (
        (upper - lower)
        / config_run[axis_length_key]
    )

    cell_fraction_t = (
        cell_count_t
        / (config_run["N"] * vol_frac)
    )

    t = simulation_time_axis(
        r,
        config_run,
    )

    return (
        float(I),
        float(D_r),
        t.copy(),
        cell_fraction_t.copy(),
    )


# ------------------------------------------------------------
# Parameter scan
# ------------------------------------------------------------

lower = 1000
upper = 1075

I_vals = np.linspace(1., 10.0, 9)
D_r_vals = np.geomspace(0.005, 0.4, 9) # [0.02, 0.067, 0.4]


pc = PlotCollector()

mse_grid = np.empty(
    (len(D_r_vals), len(I_vals))
)

cell_fraction_data = None
t_sim = None


for i_D_r, D_r in enumerate(D_r_vals):

    # One subplot = one D_r.
    # Curves within it = different I.
    cell_fraction_curves = []

    for i_I, I in enumerate(I_vals):

        I_run, D_r_run, t, cell_fraction_t = sim_run(
            config,
            f_fn,
            r_init,
            n_init,
            geometry,
            lower,
            upper,
            I,
            D_r,
        )

        if cell_fraction_data is None:
            t_sim = t.copy()
            cell_fraction_data = np.empty(
                (
                    len(D_r_vals),
                    len(I_vals),
                    len(t_sim),
                )
            )

        cell_fraction_data[
            i_D_r,
            i_I,
            :
        ] = cell_fraction_t

        mse_grid[
            i_D_r,
            i_I
        ] = calibration_mse(
            t,
            cell_fraction_t,
            calibr_spline,
        )

        cell_fraction_curves.append(
            (
                I_run,
                t,
                cell_fraction_t,
            )
        )

    pc.add(
        plot_cell_fraction_curves_ax,
        cell_fraction_curves,
        t_calibr,
        calibr_spline,
        label=rf"$D_r={D_r:g}$",
    )


# ------------------------------------------------------------
# Save scan data
# ------------------------------------------------------------

rm.save_npz(
    "lambda_optim_scan",
    I_vals=I_vals,
    D_r_vals=D_r_vals,
    t=t_sim,
    cell_fraction=cell_fraction_data,
    mse=mse_grid,
    t_calibr=t_calibr,
    cell_fraction_calibr=cell_fraction_calibr,
    lower=np.array(lower),
    upper=np.array(upper),
)


# ------------------------------------------------------------
# Cell-fraction figure: one subplot per D_r, I as curve colour
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
    name="Cell_fraction_over_time_D_r_I",
)

plt.show()


# ------------------------------------------------------------
# Discrete MSE landscape
# ------------------------------------------------------------

fig_mse, ax_mse = plt.subplots(
    figsize=(6.5, 4.8),
    constrained_layout=True,
)

plot_mse_grid_ax(
    ax_mse,
    I_vals,
    D_r_vals,
    mse_grid,
)

ax_mse.set_title(
    rf"MSE for ${lower}<y<{upper}$"
)

rm.save_plot(
    fig_mse,
    name="MSE_D_r_I",
)

plt.show()


# ------------------------------------------------------------
# Best sampled point
# ------------------------------------------------------------

i_D_r_best, i_I_best = np.unravel_index(
    np.argmin(mse_grid),
    mse_grid.shape,
)

print(
    "Best sampled parameters:"
    f"  D_r = {D_r_vals[i_D_r_best]:g},"
    f"  I = {I_vals[i_I_best]:g},"
    f"  MSE = {mse_grid[i_D_r_best, i_I_best]:.6g}"
)
