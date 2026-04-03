from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

def load_config(file_name: str) -> dict:
    path = Path(__file__).parent.parent.parent / "configs" / f"{file_name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Compute and print mean orientation
def compute_mean_n(n:np.ndarray)->np.ndarray:
    n_av = np.sum(n[-1,:], axis=-1)/n.shape[-1]
    print(f"Population averaged orientation n:\n {n_av}\n")
    return n_av

def compute_autocorr_n(n:np.ndarray)->np.ndarray:
    n_last = n[-1,:]
    autocorr = np.tensordot(n_last,n_last,axes=(1,1))/n.shape[-1]
    print(f"Population average of nn tensor:\n {autocorr}\n")
    return autocorr


# Gaussian beam light intensity field
def make_I_Gaussian_beam(config):
    sigma_beam = config["sigma_beam"]
    def I_Gaussian_beam(r): # r - positions to check field at:
        x, y = r[0,:], r[1,:]
        return np.exp(-(x**2 + y**2)/(2*sigma_beam**2))[None,:] # has to return shape (1, r.shape[1]), r.shape[1] = N
    return I_Gaussian_beam

# Unit light intensity field
def I_identity(r): # r - positions to check field at:
    return np.ones((1, r.shape[1]), dtype=float)   # has to return shape (1, r.shape[1]), r.shape[1] = N


def const_initial_conditions(config: dict, r_const: np.ndarray, n_const: np.ndarray):
    N=config["N"]
    dim=config["dim"]
    if dim!=r_const.shape[0] or dim!=n_const.shape[0]:
        raise ValueError("Wrong array dimensions")
    
    r_init = r_const[:,None]*np.ones(N)
    n_init = n_const[:,None]*np.ones(N)
    return r_init, n_init



def plot_hist_rho(
    rho: np.ndarray,
    bins: int = 40
    ):
    plt.hist(rho, bins = bins)
    plt.xlabel(fr"$\rho$")
    plt.ylabel("counts")
    plt.show()

def plot_hist_z(
    z: np.ndarray,
    bins: int = 40
    ):
    plt.hist(z, bins=40)
    plt.xlabel(fr"$z$")
    plt.ylabel("counts")
    plt.show()



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
