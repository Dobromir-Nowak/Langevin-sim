import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity, F, const_initial_conditions, const_initial_conditions_split, make_linear_grad_beam
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist, plot_hist_ax, PlotCollector
from langevin_sim.io.results import ResultsManager

from langevin_sim.physics.langevin import Langevin_sim
from langevin_sim.physics.geometry import NewCuboid



# Load plot style
parent_dir = Path(__file__).parent.parent
plt.style.use(parent_dir / "softmatter.mplstyle")

file_name = "cuboid_lin_new_bc"
config_path = Path("configs") / f"{file_name}.yaml"
config = load_config(config_path=config_path)
alpha = 0.28

f_fn = F
# I_fn = make_linear_grad_beam(config=config)

sigma_beam = 30
x0 = 50
def I_fn(r):
    x, y = r[0,:], r[1,:]
    return np.exp(-((x-x0)**2)/(2*sigma_beam**2))[None,:]






rm = ResultsManager(config_path=config_path, tag="cuboid_lin")

# Geometry
geometry = NewCuboid(config=config, alpha=alpha)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation 
save_every = 100
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=save_every)
# sim.plot_trajectories()

r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
nx, ny, nz = n[-1,0,:], n[-1,1,:], n[-1,2,:]





# Hist
r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
z_some = z[z>0.99*config["Lz"]]

# plot_hist(z, axis_label="z", bins=20)
# plot_hist(z_some, axis_label="z_some", bins=20)


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

    # optional safety against boundary overflow
    ix = np.clip(ix, 0, bins_x - 1)
    iz = np.clip(iz, 0, bins_z - 1)

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


pc = PlotCollector()
pc.add(plot_current_ax, x, z, nx, nz, config, bins_x=20, bins_z=20)
pc.add(plot_density_ax, x, z, config)
pc.add(plot_hist_ax, z, axis_label="z", bins=20)
pc.add(plot_hist_ax, x, axis_label="x", bins=20)
rm.save_plot(pc.render(), name= f"joint_fig_t={config["Nt"]*config["dt"]}")




# # 2D currents xz
# bins_x = bins_z = 20

# Lx = config["Lx"]
# Lz = config["Lz"]
# dx = Lx/bins_x
# dz = Lz/bins_z

# ix = (x/dx).astype(int) # indices of x bins
# iz = (z/dz).astype(int)

# Jx = np.zeros((bins_x,bins_z))
# Jz = np.zeros((bins_x, bins_z))
# np.add.at(Jx, (ix, iz), nx)
# np.add.at(Jz, (ix, iz), nz)
# Jx /= dx
# Jz /= dz

# x_centers = (np.arange(bins_x) + 1/2)*dx
# y_centers = (np.arange(bins_z) + 1/2)*dz
# X, Z = np.meshgrid(x_centers, y_centers, indexing='ij')

# plt.quiver(X, Z, Jx, Jz)
# plt.show()

# # Density (rho) xz
# rho = np.zeros((bins_x,bins_z))
# np.add.at(rho, (ix, iz), 1/(dx*dz))   # TODO check normalization when it becomes relevant

# plt.imshow(
#     rho.T,
#     origin='lower',
#     extent=[0, Lx, 0, Lz],
#     aspect='auto'
# )
# plt.colorbar(label='density')
# plt.show()


# # 2D currents xy

# bins_x = bins_y = 10

# Lx = config["Lx"]
# Ly = config["Ly"]
# dx = Lx/bins_x
# dy = Ly/bins_y

# ix = (x/dx).astype(int) # indices of x bins
# iy = (y/dy).astype(int)

# Jx = np.zeros((bins_x,bins_y))
# Jy = np.zeros((bins_x, bins_y))
# np.add.at(Jx, (ix, iy), nx)
# np.add.at(Jy, (ix, iy), ny)
# Jx /= dx
# Jy /= dy

# x_centers = (np.arange(bins_x) + 1/2)*dx
# y_centers = (np.arange(bins_y) + 1/2)*dy
# X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

# plt.quiver(X, Y, Jx, Jy)
# plt.show()




# # 1D currents

# bins_x = bins_y = 20

# Lx = config["Lx"]
# dx = Lx/bins_x

# ix = (x/dx).astype(int) # indices of x bins

# Jx = np.zeros(bins_x)
# np.add.at(Jx, (ix,), nx)
# Jx /= dx

# x_centers = (np.arange(bins_x)+1/2)*dx
# plt.plot(x_centers, Jx)
# plt.show()