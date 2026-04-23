from typing import Any

import matplotlib.pyplot as plt
import numpy as np

def plot_hist_rho(
    rho: np.ndarray,
    bins: int = 40, 
    show: bool = True
    ):
    fig, ax = plt.subplots()
    ax.hist(rho, bins = bins)
    ax.set_xlabel(fr"$\rho$")
    ax.set_ylabel("counts")
    if show:
        plt.show()
    return fig

def plot_density_rho(
    rho: np.ndarray,
    bins: int = 40,
    show: bool = True
    ):
    counts, edges = np.histogram(rho, bins=bins, range=(0,rho.max()))
    N_counts = len(rho)

    # bin centers and width
    rho_centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    # computing density
    density = counts / (N_counts * 2*np.pi*rho_centers*width)

    fig, ax = plt.subplots()
    ax.bar(rho_centers, density, width=width, align='center')
    ax.set_xlabel(fr"$\rho$")
    ax.set_ylabel("cell density")
    if show:
        plt.show()
    return fig

def plot_hist_x(
    x: np.ndarray,
    bins: int = 40,
    show: bool = True
    ):
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_xlabel(fr"$x$")
    ax.set_ylabel("counts")
    if show:
        plt.show()
    return fig

def plot_hist_z(
    z: np.ndarray,
    bins: int = 40,
    show: bool = True
    ):
    fig, ax = plt.subplots()
    ax.hist(z, bins=bins)
    ax.set_xlabel(fr"$z$")
    ax.set_ylabel("counts")
    if show:
        plt.show()
    return fig



def plot_trajectories(
    r_history: np.ndarray,
    n_show: int = 50,
    ax: Any | None = None,
    title: str = "Trajectories (subset)",
    start_color: str = "k",
    end_color: str = "r",
    line_alpha: float = 0.7,
    marker_alpha: float = 0.6,
    line_width: float = 1.0,
    start_size: float = 10.0,
    end_size: float = 12.0,
    show: bool = True,
    aspect_ratio: tuple[float, float, float] | None = None, 
    show_cylinder: bool = False,
    config: dict | None = None,
):
    """Plot a subset of trajectories from a history array."""
    r_history = np.asarray(r_history)
    if r_history.ndim != 3:
        raise ValueError(
            f"r_history must have shape (n_times, dim, n_particles); got {r_history.shape}"
        )

    n_times, dim, n_particles = r_history.shape
    if n_times == 0 or n_particles == 0:
        raise ValueError("r_history must contain at least one time point and one particle")
    if dim not in (2, 3):
        raise ValueError(f"Only 2D or 3D trajectory plots are supported; got dim={dim}")
    if show_cylinder:
        if dim != 3:
            raise ValueError("Cylinder plotting only supported for 3D")
        if config is None:
            raise ValueError("cylinder_config must be provided when show_cylinder=True")
        required_keys = ["R_cylinder", "zmin", "zmax"]
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Missing '{k}' in cylinder_config")

    n_show = min(max(int(n_show), 1), n_particles)
    idx = np.linspace(0, n_particles - 1, n_show, dtype=int)

    created_figure = ax is None
    if ax is None:
        if dim == 2:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if show_cylinder:
        R = config["R_cylinder"]
        zmin = config["zmin"]
        zmax = config["zmax"]

        theta = np.linspace(0, 2*np.pi, 60)
        z = np.linspace(zmin, zmax, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x = R * np.cos(theta_grid)
        y = R * np.sin(theta_grid)

        ax.plot_surface(
            x, y, z_grid,
            alpha=0.05,          # very faint
            linewidth=0,
            antialiased=True
        )


    for i in idx:
        if dim == 2:
            ax.plot(
                r_history[:, 0, i],
                r_history[:, 1, i],
                lw=line_width,
                alpha=line_alpha,
            )
            ax.scatter(
                r_history[0, 0, i],
                r_history[0, 1, i],
                s=start_size,
                c=start_color,
                alpha=marker_alpha,
            )
            ax.scatter(
                r_history[-1, 0, i],
                r_history[-1, 1, i],
                s=end_size,
                c=end_color,
                alpha=marker_alpha,
            )
        else:
            ax.plot(
                r_history[:, 0, i],
                r_history[:, 1, i],
                r_history[:, 2, i],
                lw=line_width,
                alpha=line_alpha,
            )
            ax.scatter(
                r_history[0, 0, i],
                r_history[0, 1, i],
                r_history[0, 2, i],
                s=start_size,
                c=start_color,
                alpha=marker_alpha,
            )
            ax.scatter(
                r_history[-1, 0, i],
                r_history[-1, 1, i],
                r_history[-1, 2, i],
                s=end_size,
                c=end_color,
                alpha=marker_alpha,
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if dim == 2:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    else:
        ax.set_zlabel("z")
        ax.set_box_aspect(aspect_ratio)
    if show and created_figure:
        plt.show()

    return fig, ax
