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

def plot_current_ax(ax, x, z, nx, nz, config, bins_x=20, bins_z=20):

    Lx = config["Lx"]
    Lz = config["Lz"]

    dx = Lx / bins_x
    dz = Lz / bins_z

    ix = (x / dx).astype(int)
    iz = (z / dz).astype(int)

    Jx = np.zeros((bins_x, bins_z))
    Jz = np.zeros((bins_x, bins_z))

    np.add.at(Jx, (ix, iz), nx)
    np.add.at(Jz, (ix, iz), nz)

    Jx /= dx
    Jz /= dz

    x_centers = (np.arange(bins_x) + 0.5) * dx
    z_centers = (np.arange(bins_z) + 0.5) * dz

    X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")

    ax.quiver(X, Z, Jx, Jz)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)



def plot_density_ax(ax, x, z, config, bins_x=20, bins_z=20, cmap="viridis"):
    Lx = config["Lx"]
    Lz = config["Lz"]

    dx = Lx / bins_x
    dz = Lz / bins_z

    ix = (x / dx).astype(int)
    iz = (z / dz).astype(int)

    rho = np.zeros((bins_x, bins_z))

    np.add.at(rho, (ix, iz), 1 / (dx * dz))

    im = ax.imshow(
        rho.T,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="auto",
        cmap=cmap
    )

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Lz)


def plot_n_correlation(ax, n: np.ndarray, config: dict, absolute = False):
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
        ax.set_ylabel(r"$|\langle \mathbf{n}(t)\cdot\mathbf{n}(0)\rangle|$")
    else:
        ax.plot(t_lin, mean_dot_prod)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\langle \mathbf{n}(t)\cdot\mathbf{n}(0)\rangle$")
