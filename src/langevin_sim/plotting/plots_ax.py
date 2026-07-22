from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class PlotCollector:
    def __init__(self):
        self.jobs = []

    def add(self, plot_fn, *args, label=None, **kwargs):
        self.jobs.append({
            "fn": plot_fn,
            "args": args,
            "kwargs": kwargs,
            "label": label
        })

    def render(self, layout="grid", ncols=2, sharex=False, sharey=False, show=True):
        n = len(self.jobs)

        if layout == "grid":
            nrows = np.ceil(n / ncols).astype(int)
        elif layout == "column":
            ncols, nrows = 1, n
        elif layout == "row":
            ncols, nrows = n, 1
        else:
            raise ValueError("unknown layout")

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5*ncols, 4*nrows),
                                 squeeze=False,
                                 sharex=sharex, sharey=sharey)

        axes = axes.flatten()

        for i, job in enumerate(self.jobs):
            ax = axes[i]
            job["fn"](ax, *job["args"], **job["kwargs"])

            if job["label"] is not None:
                ax.set_title(job["label"])

        # remove unused axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        if show:
            plt.show()

        return fig

def plot_ax(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str | None = None,
    y_label: str | None = None
    ):
    if x_label is not None:
        ax.set_xlabel(fr"{x_label}")
    if y_label is not None:
        ax.set_ylabel(fr"{y_label}")
    ax.plot(x,y)

def plot_hist_ax(
    ax, 
    x: np.ndarray,
    axis_label: str,
    bins: int = 40,
    label: str | None = None,
    bin_stats: bool = False,
    show_count_fluct: bool = False
    ):

    counts, bin_edges, _ = ax.hist(x, bins=bins)

    ax.set_xlabel(fr"${axis_label}$")
    ax.set_ylabel("counts")
    if label is not None:
        ax.set_title(label)
    if bin_stats:
        mean = np.mean(counts)
        s = np.sqrt(mean) # uncertainty due to binning for a uniform distribution
        ax.text(
            0.95, 0.95,
            fr"$\mu={mean:.0f},\ \sigma={s:.0f}$",
            transform=ax.transAxes,
            ha="right", va="top"
        )
    if show_count_fluct:
        ax.axhline(float(mean), linestyle="--", color="r", linewidth=1., label =r"$\mu$")
        ax.axhline(float(mean + s), linestyle=":", color="r",linewidth=1., label=r"$\mu\pm\sigma$")
        ax.axhline(float(mean - s), linestyle=":", color="r",linewidth=1.)
        ax.legend(frameon=False)


def plot_density_ax(ax, r, config, axis_i, axis_j, bins_xi=20, bins_xj=20, cmap="viridis", cbar_label=r"$n$", plot_label="Cell density"):
    
    if axis_i not in (0,1,2):
        raise ValueError
    if axis_j not in (0,1,2):
        raise ValueError

    L = np.array([config["Lx"], config["Ly"], config["Lz"]])
    dx_i = L[axis_i]/bins_xi
    dx_j = L[axis_j]/bins_xj

    x_i = r[axis_i,:]
    x_j = r[axis_j,:]

    x_axis_names = [r"$x$", r"$y$", r"$z$"]
    

    ix_i = (x_i / dx_i).astype(int)
    ix_j = (x_j / dx_j).astype(int)

    rho = np.zeros((bins_xi, bins_xj))

    np.add.at(rho, (ix_i, ix_j), 1/(dx_i * dx_j))

    im = ax.imshow(
        rho.T,
        origin="lower",
        extent=[0, L[axis_i], 0, L[axis_j]],
        aspect="auto",
        cmap=cmap
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlim(0, L[axis_i])
    ax.set_ylim(0, L[axis_j])
    ax.set_xlabel(x_axis_names[axis_i])
    ax.set_ylabel(x_axis_names[axis_j])
    ax.set_title(plot_label)


def plot_current_ax(
    ax,
    r,
    n,
    config,
    axis_i,
    axis_j,
    bins_xi=20,
    bins_xj=20,
    plot_label="Cell current density",
    quiver_scale=None,
    L=None
):
    if axis_i not in (0, 1, 2):
        raise ValueError("axis_i must be 0, 1, or 2")
    if axis_j not in (0, 1, 2):
        raise ValueError("axis_j must be 0, 1, or 2")
    if axis_i == axis_j:
        raise ValueError("axis_i and axis_j must be different")

    v0 = config["v0"]

    if L is None:
        L = np.array([config["Lx"], config["Ly"], config["Lz"]])
    else:
        L=L
    dx_i = L[axis_i] / bins_xi
    dx_j = L[axis_j] / bins_xj

    x_i = r[axis_i, :]
    x_j = r[axis_j, :]

    n_i = n[axis_i, :]
    n_j = n[axis_j, :]

    axis_names = [r"$x$", r"$y$", r"$z$"]

    ix_i = (x_i / dx_i).astype(int)
    ix_j = (x_j / dx_j).astype(int)

    J_i = np.zeros((bins_xi, bins_xj))
    J_j = np.zeros((bins_xi, bins_xj))

    np.add.at(J_i, (ix_i, ix_j), v0 * n_i)
    np.add.at(J_j, (ix_i, ix_j), v0 * n_j)

    J_i /= dx_i * dx_j
    J_j /= dx_i * dx_j

    x_i_centers = (np.arange(bins_xi) + 0.5) * dx_i
    x_j_centers = (np.arange(bins_xj) + 0.5) * dx_j

    X_i, X_j = np.meshgrid(x_i_centers, x_j_centers, indexing="ij")

    ax.quiver(
        X_i,
        X_j,
        J_i,
        J_j,
        scale=quiver_scale,
    )

    ax.set_xlim(0, L[axis_i])
    ax.set_ylim(0, L[axis_j])
    ax.set_xlabel(axis_names[axis_i])
    ax.set_ylabel(axis_names[axis_j])
    ax.set_title(plot_label)





def plot_n_correlation(ax, n: np.ndarray, config: dict, absolute = False, log=False):
    dt = config["dt"]
    save_every = config["save_every"]
    Dt = save_every * dt
    nt = n.shape[0]
    t_lin = Dt * np.arange(nt)

    n_init = n[0, :, :][None, :, :]
    dot_prod = np.sum(n * n_init, axis=1)
    mean_dot_prod = np.mean(dot_prod, axis=1)
    if absolute:
        ax.plot(t_lin, np.abs(mean_dot_prod))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$|\langle \hat{\mathbf{n}}(t)\cdot \hat{\mathbf{n}}(0)\rangle|$")
    else:
        ax.plot(t_lin, mean_dot_prod)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\langle \hat{\mathbf{n}}(t)\cdot\hat{\mathbf{n}}(0)\rangle$")
    if log: 
        ax.set_yscale('log')


def plot_hist_lin_ax(ax, r, config, axis=None, bins=20, n_plots=5, t_label=True, log=False):

    if axis not in (0,1,2):
        raise ValueError
    x = r[:,axis,:]

    save_every = config["save_every"]
    Nt = config["Nt"]
    if n_plots > Nt//save_every:
        raise ValueError("n_plots too large")
    t_idx_list = np.arange(n_plots)
    x = np.copy(x[t_idx_list,:])

    t_list = config["dt"]*t_idx_list*save_every

    axis_names = [r"$x$", r"$y$", r"$z$"]
    side_names = ["Lx", "Ly", "Lz"]

    L = config[side_names[axis]]
    dx = L / bins
    ix = (x / dx).astype(int)
    x_bins = (np.arange(bins)+1/2)*dx

    # binning
    counts = np.zeros((n_plots,bins))
    binning_indices = np.arange(n_plots)[:,None]*np.ones(ix.shape).astype(int)
    np.add.at(counts, (binning_indices, ix), 1)

    # making plots
    for idx_t, t in enumerate(t_list):
        if t_label:
            ax.plot(x_bins, counts[idx_t,:], marker='.', markersize=2, markerfacecolor='black', markeredgecolor='black', label=fr"$t={t:.0f}$")
        else:
            ax.plot(x_bins, counts[idx_t,:], marker='.', markersize=2, markerfacecolor='black', markeredgecolor='black', label=fr"$t={t:.0f}$")
        ax.set_xlabel(axis_names[axis])
        ax.set_ylabel("counts")
        if log: 
            ax.set_yscale('log')
        ax.legend()


def plot_current_magnitude_and_direction_ax(
    ax, x, z, nx, nz, config,
    bins_x=20, bins_z=20,
    cmap="viridis"
):
    v0 = config["v0"]

    Lx = config["Lx"]
    Lz = config["Lz"]

    dx = Lx / bins_x
    dz = Lz / bins_z

    ix = np.clip((x / dx).astype(int), 0, bins_x - 1)
    iz = np.clip((z / dz).astype(int), 0, bins_z - 1)

    Jx = np.zeros((bins_x, bins_z))
    Jz = np.zeros((bins_x, bins_z))

    np.add.at(Jx, (ix, iz), v0*nx)
    np.add.at(Jz, (ix, iz), v0*nz)

    Jx /= (dx * dz)
    Jz /= (dx * dz)

    # magnitude
    J = np.sqrt(Jx**2 + Jz**2)

    # normalized direction field
    eps = 1e-12
    U = Jx / (J + eps)
    V = Jz / (J + eps)

    x_centers = (np.arange(bins_x) + 0.5) * dx
    z_centers = (np.arange(bins_z) + 0.5) * dz

    X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")

    # background magnitude field
    im = ax.imshow(
        J.T,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="auto",
        cmap=cmap
    )

    plt.colorbar(im, ax=ax, label=r"$|\mathbf{J}|$")

    # unit direction vectors
    ax.quiver(
        X, Z,
        U, V,
        color="white",
        pivot="mid",
        scale=30
    )

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$z$")
    ax.set_title("Current magnitude and direction")




def plot_mean_polarization_ax(
    ax, x, z, nx, nz, config,
    bins_x=20, bins_z=20,
    cmap="viridis"
):

    Lx = config["Lx"]
    Lz = config["Lz"]

    dx = Lx / bins_x
    dz = Lz / bins_z

    ix = np.clip((x / dx).astype(int), 0, bins_x - 1)
    iz = np.clip((z / dz).astype(int), 0, bins_z - 1)

    Px = np.zeros((bins_x, bins_z))
    Pz = np.zeros((bins_x, bins_z))

    bin_counts = np.zeros((bins_x, bins_z))

    np.add.at(Px, (ix, iz), nx)
    np.add.at(Pz, (ix, iz), nz)

    np.add.at(bin_counts, (ix, iz), 1)

    zero_mask = bin_counts == 0
    Px[~zero_mask]/=bin_counts[~zero_mask]
    Pz[~zero_mask]/=bin_counts[~zero_mask]

    # magnitude
    P = np.sqrt(Px**2 + Pz**2) # Px # Pz

    x_centers = (np.arange(bins_x) + 0.5) * dx
    z_centers = (np.arange(bins_z) + 0.5) * dz

    X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")

    # background magnitude field
    im = ax.imshow(
        P.T,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="auto",
        cmap=cmap
    )

    plt.colorbar(im, ax=ax, label=r"$|\mathbf{P}|$")
    ax.set_title("Mean polarization")



def plot_density_rho_ax(
    ax,
    rho: np.ndarray,
    bins: int = 40
    ):
    counts, edges = np.histogram(rho, bins=bins, range=(0,rho.max()))
    N_counts = len(rho)

    # bin centers and width
    rho_centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    # computing density
    rho2 = edges[1:]
    rho1 = edges[:-1]
    areas = np.pi*(rho2**2 - rho1**2)
    density = counts/ (N_counts*areas)
    # # alternative approximation
    # density = counts / (N_counts * 2*np.pi*rho_centers*width)

    ax.bar(rho_centers, density, width=width, align='center')
    ax.set_xlabel(fr"$\rho$")
    ax.set_ylabel("cell density")



# computing {axis} component of current density after binning over coordinate {axis}
def plot_current_lin_ax(ax, r, n, config, par_vals: np.ndarray | None = None, axis_r=None, axis_n=None, bins=20, log=False):

    
    v0 = config["v0"]
    L = np.array([config["Lx"], config["Ly"], config["Lz"]])
    dx = L[axis_r]/bins
    n_plots = r.shape[0]

    if axis_r not in (0,1,2):
        raise ValueError
    if axis_n not in (0,1,2):
        raise ValueError

    x = r[:,axis_r,:]
    nx = n[:,axis_n,:]
    
    ix = (x / dx).astype(int)
    jx = np.zeros((n_plots, bins))

    binning_indices = np.arange(n_plots)[:,None]*np.ones(ix.shape[1]).astype(int)
    np.add.at(jx, (binning_indices, ix), v0*nx)
    jx/= np.prod(L) / L[axis_r]*dx   #  = L_i * L_j * dx # TODO check if axis_r or axis_n

    # making plots
    x_axis_names = [r"$x$", r"$y$", r"$z$"]
    y_axis_names = [r"$j_x$", r"$j_y$", r"$j_z$"]
    x_bins = (np.arange(bins)+1/2)*dx


    for idx in range(n_plots):
        if par_vals is None:
            ax.plot(x_bins, jx[idx,:], marker='.', markersize=2, markerfacecolor='black', markeredgecolor='black')
        else:
            par_vals_deg = (180/np.pi) * par_vals
            ax.plot(x_bins, jx[idx,:], marker='.', markersize=2, markerfacecolor='black', markeredgecolor='black', label=fr"$\theta={par_vals_deg[idx]:.0f}^\circ$")
        ax.set_xlabel(x_axis_names[axis_r])
        ax.set_ylabel(y_axis_names[axis_n])
        if log: 
            ax.set_yscale('log')
        ax.legend()




def plot_trajectories_2d_ax(
    ax,
    r_history: np.ndarray,
    excluded_axis: int | None = None,
    n_show: int = 50,
    config: dict | None = None,
    axis_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    periodic_axes: tuple[int, ...] | None = None,
    break_periodic_jumps: bool = True,
    start_color: str = "k",
    end_color: str = "r",
    line_alpha: float = 0.7,
    marker_alpha: float = 0.7,
    line_width: float = 1.0,
    start_size: float = 10.0,
    end_size: float = 12.0,
    equal_aspect: bool = True,
    grid: bool = True,
):
    """Plot 2D trajectories from a 2D history or a projection of a 3D history.

    ``r_history`` must have shape ``(n_times, dim, n_particles)``. For 3D
    histories, ``excluded_axis`` selects the coordinate omitted from the plot:
    0 -> y-z, 1 -> x-z, 2 -> x-y.

    For wrapped periodic coordinates, set ``periodic_axes`` to the corresponding
    original coordinate indices. Discontinuities larger than half the domain
    length are then broken instead of being drawn across the entire plot.
    """
    r_history = np.asarray(r_history)
    if r_history.ndim != 3:
        raise ValueError(
            "r_history must have shape (n_times, dim, n_particles); "
            f"got {r_history.shape}"
        )

    n_times, dim, n_particles = r_history.shape
    if n_times == 0 or n_particles == 0:
        raise ValueError("r_history must contain at least one time and one particle")

    if dim == 2:
        if excluded_axis is not None:
            raise ValueError("excluded_axis must be None for a 2D history")
        plot_axes = (0, 1)
    elif dim == 3:
        if excluded_axis not in (0, 1, 2):
            raise ValueError("excluded_axis must be 0, 1, or 2 for a 3D history")
        plot_axes = tuple(axis for axis in range(3) if axis != excluded_axis)
    else:
        raise ValueError(f"Only 2D and 3D histories are supported; got dim={dim}")

    axis_names = ("x", "y", "z")
    side_names = ("Lx", "Ly", "Lz")

    if periodic_axes is None:
        # This matches the current Cuboid implementation: x and y are periodic
        # when bc_type == "custom_bb+pbc", while z uses custom bounce-back.
        periodic_axes = (0, 1) if config and config.get("bc_type") == "custom_bb+pbc" else ()
    invalid_periodic_axes = set(periodic_axes) - set(range(dim))
    if invalid_periodic_axes:
        raise ValueError(f"Invalid periodic axes: {sorted(invalid_periodic_axes)}")

    n_show = min(max(int(n_show), 1), n_particles)
    particle_indices = np.linspace(0, n_particles - 1, n_show, dtype=int)
    axis_0, axis_1 = plot_axes

    def plotted_coordinates(particle: int):
        coordinate_0 = r_history[:, axis_0, particle]
        coordinate_1 = r_history[:, axis_1, particle]

        if not break_periodic_jumps or n_times < 2:
            return coordinate_0, coordinate_1

        breaks = np.zeros(n_times - 1, dtype=bool)
        for axis, coordinate in ((axis_0, coordinate_0), (axis_1, coordinate_1)):
            if axis in periodic_axes:
                if config is None or side_names[axis] not in config:
                    raise ValueError(
                        f"config['{side_names[axis]}'] is required to break "
                        f"periodic jumps along {axis_names[axis]}"
                    )
                domain_length = float(config[side_names[axis]])
                breaks |= np.abs(np.diff(coordinate)) > 0.5 * domain_length

        if not np.any(breaks):
            return coordinate_0, coordinate_1

        insertion_indices = np.flatnonzero(breaks) + 1
        return (
            np.insert(coordinate_0, insertion_indices, np.nan),
            np.insert(coordinate_1, insertion_indices, np.nan),
        )

    for particle in particle_indices:
        coordinate_0, coordinate_1 = plotted_coordinates(particle)
        ax.plot(
            coordinate_0,
            coordinate_1,
            lw=line_width,
            alpha=line_alpha,
        )
        ax.scatter(
            r_history[0, axis_0, particle],
            r_history[0, axis_1, particle],
            s=start_size,
            c=start_color,
            alpha=marker_alpha,
        )
        ax.scatter(
            r_history[-1, axis_0, particle],
            r_history[-1, axis_1, particle],
            s=end_size,
            c=end_color,
            alpha=marker_alpha,
        )

    if axis_limits is None and config is not None:
        if all(side_names[axis] in config for axis in plot_axes):
            axis_limits = ((0.0, float(config[side_names[axis_0]])),
                           (0.0, float(config[side_names[axis_1]])),
            )

    if axis_limits is not None:
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        
    ax.set_xlabel(fr"${axis_names[axis_0]}$")
    ax.set_ylabel(fr"${axis_names[axis_1]}$")

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, alpha=0.2)       