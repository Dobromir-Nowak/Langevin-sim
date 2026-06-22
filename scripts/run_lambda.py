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


lower = 500
upper = 1500
axis = 1 # y 
I_min = 0.5
I_max = 1.5

def base_fn(r:np.ndarray): 
    x_i = r[axis,:][None,:] 
    return (I_max - I_min)*(x_i - lower)/(upper - lower) + I_min

I_fn = make_gated_intensity(base_fn, axis=axis, lower=lower, upper=upper)

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


# # z histograms as a single plot
# pc = PlotCollector()
# pc.add(plot_hist_lin_ax, r, config, axis=2, t_label=True)
# pc.add(plot_hist_lin_ax, r, config, axis=0, t_label=False)
# rm.save_plot(pc.render(), name= f"test_fig")


# # Orientation decorrelation
# pc = PlotCollector()
# pc.add(plot_n_correlation, n, config, absolute=True)
# pc.add(plot_n_correlation, n, config, absolute=False)
# rm.save_plot(pc.render(), name = "orientation_decorelation")

# # Gifs
# rm.save_gif(make_gif(x, z, nx, nz, plot_func=plot_current_ax, config=config, show=False, fps=10), name="current", save_fps=10)
# rm.save_gif(make_gif(x, plot_func=plot_hist_ax, axis_label="x", show=False, fps=10), name="hist", save_fps=10)

sim.plot_trajectories(aspect_ratio=[config["Lx"],config["Ly"],config["Lz"]])

x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
nx, ny, nz = n[-1,0,:], n[-1,1,:], n[-1,2,:]
pc = PlotCollector()
bins=20

# pc.add(plot_current_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z) # TODO reimplement
pc.add(plot_density_ax, r[-1,:], config, axis_i=0, axis_j=2, bins_xi=bins, bins_xj=bins) # xz
pc.add(plot_density_ax, r[-1,:], config, axis_i=0, axis_j=1, bins_xi=bins, bins_xj=bins) # xy
pc.add(plot_current_ax, r[-1,:], n[-1,:], config, axis_i=0, axis_j=2, bins_xi=bins, bins_xj=bins)


# pc.add(plot_hist_ax, z, axis_label=r"z", bins=20)
# pc.add(plot_hist_ax, x, axis_label=r"x", bins=20)
# pc.add(plot_current_magnitude_and_direction_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z)
# pc.add(plot_mean_polarization_ax, x, z, nx, nz, config, bins_x=bins_x, bins_z=bins_z)
rm.save_plot(pc.render(), name= f"joint_fig_t={config['Nt']*config['dt']}")
























r_last = r[-1, :]   # shape (dim, N)
n_last = n[-1, :]   # shape (dim, N)

mask = (lower < r_last[1, :]) & (r_last[1, :] < upper)

if not np.any(mask):
    raise ValueError(rf"No particles satisfy ${lower} < y < {upper}$")

r_f = r_last[:, mask]
n_f = n_last[:, mask]

v0 = config["v0"]

fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# ------------------------------------------------------------
# 1. density in xz, binned over filtered x/z ranges
# ------------------------------------------------------------
x = r_f[0, :]
z = r_f[2, :]

rho_xz, x_edges, z_edges = np.histogram2d(
    x,
    z,
    bins=[bins, bins],
    range=[[x.min(), x.max()], [z.min(), z.max()]],
)

dx = x_edges[1] - x_edges[0]
dz = z_edges[1] - z_edges[0]
rho_xz = rho_xz / (dx * dz)

im0 = axs[0].imshow(
    rho_xz.T,
    origin="lower",
    extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
    aspect="auto",
    cmap="viridis",
)

axs[0].set_xlabel(r"$x$")
axs[0].set_ylabel(r"$z$")
axs[0].set_title(rf"Density in $xz$, ${lower} < y < {upper}$")
fig.colorbar(im0, ax=axs[0], label=r"$n$")


# ------------------------------------------------------------
# 2. density in xy, binned over filtered x/y ranges
# ------------------------------------------------------------
x = r_f[0, :]
y = r_f[1, :]

rho_xy, x_edges, y_edges = np.histogram2d(
    x,
    y,
    bins=[bins, bins],
    range=[[x.min(), x.max()], [y.min(), y.max()]],
)

dx = x_edges[1] - x_edges[0]
dy = y_edges[1] - y_edges[0]
rho_xy = rho_xy / (dx * dy)

im1 = axs[1].imshow(
    rho_xy.T,
    origin="lower",
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    aspect="auto",
    cmap="viridis",
)

axs[1].set_xlabel(r"$x$")
axs[1].set_ylabel(r"$y$")
axs[1].set_title(rf"Density in $xy$, ${lower} < y < {upper}$")
fig.colorbar(im1, ax=axs[1], label=r"$n$")


# ------------------------------------------------------------
# 3. current density in xz, binned over filtered x/z ranges
# ------------------------------------------------------------
x = r_f[0, :]
z = r_f[2, :]

nx = n_f[0, :]
nz = n_f[2, :]

Jx, x_edges, z_edges = np.histogram2d(
    x,
    z,
    bins=[bins, bins],
    range=[[x.min(), x.max()], [z.min(), z.max()]],
    weights=v0 * nx,
)

Jz, _, _ = np.histogram2d(
    x,
    z,
    bins=[bins, bins],
    range=[[x.min(), x.max()], [z.min(), z.max()]],
    weights=v0 * nz,
)

dx = x_edges[1] - x_edges[0]
dz = z_edges[1] - z_edges[0]

Jx = Jx / (dx * dz)
Jz = Jz / (dx * dz)

x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

X, Z = np.meshgrid(x_centers, z_centers, indexing="ij")

axs[2].quiver(X, Z, Jx, Jz)

axs[2].set_xlim(x_edges[0], x_edges[-1])
axs[2].set_ylim(z_edges[0], z_edges[-1])
axs[2].set_xlabel(r"$x$")
axs[2].set_ylabel(r"$z$")
axs[2].set_title(rf"Current in $xz$, ${lower} < y < {upper}$")

plt.show()


