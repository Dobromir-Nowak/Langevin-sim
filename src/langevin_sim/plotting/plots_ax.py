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

def plot_current_ax(ax, x, z, nx, nz, config, bins_x=20, bins_z=20, xlabel=r"$x$", ylabel=r"$z$", plot_label="Cell current density"):

    v0 = config["v0"]

    Lx = config["Lx"]
    Lz = config["Lz"]

    dx = Lx / bins_x
    dz = Lz / bins_z

    ix = (x / dx).astype(int)
    iz = (z / dz).astype(int)

    Jx = np.zeros((bins_x, bins_z))
    Jz = np.zeros((bins_x, bins_z))

    np.add.at(Jx, (ix, iz), v0*nx)
    np.add.at(Jz, (ix, iz), v0*nz)

    Jx /= (dx*dz)
    Jz /= (dx*dz)

    x_centers = (np.arange(bins_x) + 0.5) * dx
    z_centers = (np.arange(bins_z) + 0.5) * dz

    X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")

    ax.quiver(X, Z, Jx, Jz)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(plot_label)

def plot_density_ax(ax, x, z, config, bins_x=20, bins_z=20, cmap="viridis", xlabel = r"$x$", ylabel = r"$z$", cbar_label=r"$n$", plot_label="Cell density"):
    Lx = config["Lx"]
    Lz = config["Lz"]

    dx = Lx / bins_x
    dz = Lz / bins_z

    ix = (x / dx).astype(int)
    iz = (z / dz).astype(int)

    rho = np.zeros((bins_x, bins_z))

    np.add.at(rho, (ix, iz), 1/(dx * dz))

    im = ax.imshow(
        rho.T,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="auto",
        cmap=cmap
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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



# TODO adjust to the use case  -- computing {axis} component of current density after binning over coordinate {axis}
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