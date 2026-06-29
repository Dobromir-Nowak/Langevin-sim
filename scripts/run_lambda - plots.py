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

file_name = "cuboid_lambda"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)

f_fn = F

# Creating the spline
import pandas as pd
from scipy.interpolate import PchipInterpolator
csv_path = Path("configs/mcwhl5_spectrum_digitized_from_plot_5nm.csv")
spec = pd.read_csv(csv_path)

lam_data = spec["wavelength_nm"].to_numpy()
S_data = spec["normalized_intensity"].to_numpy()

lam_normalized = (lam_data - lam_data.min())/(lam_data.max() - lam_data.min())

S = PchipInterpolator(lam_normalized, S_data, extrapolate=False)

# Creating the intensity profile

lower = 500
upper = 1500
axis = 1 # y 
I_multiplier = 10. # peak intensity

def base_fn_spline(r:np.ndarray): 
    x_i = r[axis,:][None,:]
    x_i_scalled = (x_i - lower)/(upper - lower)
    x_i_scalled = 1 - x_i_scalled # flipping to match lambda from 800 to 400
    return I_multiplier*S(x_i_scalled)   # S - 1d spline

I_fn = make_gated_intensity(base_fn_spline, axis=axis, lower=lower, upper=upper)

rm = ResultsManager(config_path=config_path, tag=file_name)

# Geometry
geometry = Cuboid(config=config)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)


# Run simulation
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=config["save_every"])

r, n = results["r"], results["n"]

x, y, z = r[:,0,:], r[:,1,:], r[:,2,:]
nx, ny, nz = n[:,0,:], n[:,1,:], n[:,2,:]











from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def simulation_time_axis(r, config, t_offset=0.0):
    """
    Time axis for saved simulation frames.

    r shape: (T, dim, N)
    """
    save_every = int(config.get("save_every", 1))
    dt = float(config["dt"])
    return t_offset + np.arange(r.shape[0]) * save_every * dt


def wavelength_from_y(y, lower, upper, lam_min=400.0, lam_max=800.0):
    """
    Matches your intensity construction:
        y = lower -> lam_max
        y = upper -> lam_min
    """
    y_scaled = (y - lower) / (upper - lower)
    return lam_max - y_scaled * (lam_max - lam_min)


def binned_counts_over_time(
    r,
    slice_axis,
    bins=20,
    slice_range=None,
    restrict_axis=None,
    restrict_range=None,
    normalize="all",
):
    """
    Returns binned particle counts as a line-plot-ready array.

    Parameters
    ----------
    r : ndarray, shape (T, dim, N)
        Saved particle positions.
    slice_axis : int
        Axis along which slices/bins are made.
        0 -> x, 1 -> y, 2 -> z.
    bins : int
        Number of spatial bins, e.g. 20.
    slice_range : tuple or None
        Spatial range to bin over. If None, inferred from selected data.
    restrict_axis : int or None
        Optional axis used only for filtering.
        Example: restrict_axis=1 with restrict_range=(lower, upper)
        means "only particles with lower < y < upper".
    restrict_range : tuple or None
        Range for optional filtering.
    normalize : {"all", "selected", "none"}
        "all"      : counts / total particle number N.
        "selected" : counts / number of particles passing filter at each time.
        "none"     : raw counts.

    Returns
    -------
    values : ndarray, shape (T, bins)
        Time series for each bin.
    centers : ndarray, shape (bins,)
        Bin centers.
    edges : ndarray, shape (bins + 1,)
        Bin edges.
    """

    if r.ndim != 3:
        raise ValueError("r must have shape (T, dim, N)")

    T, dim, N = r.shape

    coords = r[:, slice_axis, :]

    mask = np.ones((T, N), dtype=bool)

    if restrict_axis is not None:
        if restrict_range is None:
            raise ValueError("restrict_range must be provided when restrict_axis is not None")

        lo, hi = restrict_range
        rr = r[:, restrict_axis, :]
        mask &= (rr >= lo) & (rr <= hi)

    if slice_range is None:
        selected = coords[mask]
        if selected.size == 0:
            raise ValueError("No particles found in the requested restricted region")
        slice_range = (selected.min(), selected.max())

    edges = np.linspace(slice_range[0], slice_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = np.zeros((T, bins), dtype=float)
    selected_counts = np.zeros(T, dtype=float)

    for it in range(T):
        m = mask[it]
        selected_counts[it] = np.sum(m)
        counts[it], _ = np.histogram(coords[it, m], bins=edges)

    if normalize == "all":
        values = counts / N

    elif normalize == "selected":
        values = np.divide(
            counts,
            selected_counts[:, None],
            out=np.zeros_like(counts),
            where=selected_counts[:, None] > 0,
        )

    elif normalize == "none":
        values = counts

    else:
        raise ValueError("normalize must be one of: 'all', 'selected', 'none'")

    return values, centers, edges


def plot_binned_line_slices_ax(
    ax,
    r,
    config,
    slice_axis,
    bins=20,
    slice_range=None,
    restrict_axis=None,
    restrict_range=None,
    normalize="all",
    cmap="gray",
    color_values_fn=None,
    colorbar_label=None,
    title=None,
    ylabel=r"Normalized cell density $N_k(t)$",
    t_offset=0.0,
    lw=1.2,
    alpha=0.95,
):
    """
    Line plot of binned densities over time.
    Each spatial bin becomes one line.
    """

    t = simulation_time_axis(r, config, t_offset=t_offset)

    values, centers, edges = binned_counts_over_time(
        r,
        slice_axis=slice_axis,
        bins=bins,
        slice_range=slice_range,
        restrict_axis=restrict_axis,
        restrict_range=restrict_range,
        normalize=normalize,
    )

    if color_values_fn is None:
        color_values = centers
    else:
        color_values = color_values_fn(centers)

    norm = Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for k in range(bins):
        ax.plot(
            t,
            values[:, k],
            color=sm.to_rgba(color_values[k]),
            lw=lw,
            alpha=alpha,
        )

    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if colorbar_label is not None:
        ax.figure.colorbar(sm, ax=ax, label=colorbar_label)

    return values, centers, edges







bins = 10

# ------------------------------------------------------------
# 1. Vertical-slice analogue:
#    bins in x, but only for particles in the illuminated/central y-strip
#    lower < y < upper.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

plot_binned_line_slices_ax(
    ax,
    r,
    config,
    slice_axis=0,                     # x-bins: vertical slices
    bins=bins,
    restrict_axis=1,                  # only central y-region
    restrict_range=(lower, upper),
    normalize="all",                  # counts / total number of particles
    cmap="gray",
    colorbar_label=r"$x$",
    title=rf"Vertical slices, ${lower}<y<{upper}$",
)

ax.set_ylim(bottom=0)

rm.save_plot(fig, name=f"vertical_slices_central_y_{lower}_{upper}")


# ------------------------------------------------------------
# 2. Horizontal/wavelength-slice analogue:
#    bins directly in y over lower < y < upper.
#    Color each line by the wavelength corresponding to that y-bin.
# ------------------------------------------------------------
lam_min = float(lam_data.min())
lam_max = float(lam_data.max())

fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

plot_binned_line_slices_ax(
    ax,
    r,
    config,
    slice_axis=1,                     # y-bins: horizontal/wavelength slices
    bins=bins,
    slice_range=(lower, upper),       # central illuminated part only
    normalize="all",
    cmap="jet",
    color_values_fn=lambda yy: wavelength_from_y(
        yy,
        lower=lower,
        upper=upper,
        lam_min=lam_min,
        lam_max=lam_max,
    ),
    colorbar_label=r"Wavelength [nm]",
    title=rf"Horizontal slices, ${lower}<y<{upper}$",
)

ax.set_ylim(bottom=0)

rm.save_plot(fig, name=f"horizontal_wavelength_slices_y_{lower}_{upper}")