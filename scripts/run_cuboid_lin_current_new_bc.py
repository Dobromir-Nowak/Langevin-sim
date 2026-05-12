import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from langevin_sim.utils.compute import I_identity, F, const_initial_conditions, const_initial_conditions_split, make_linear_grad_beam
from langevin_sim.utils.other import load_config
from langevin_sim.plotting.plots import plot_hist
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
I_fn = make_linear_grad_beam(config=config)

rm = ResultsManager(config_path=config_path, tag="cuboid_lin")

# Geometry
geometry = NewCuboid(config=config, alpha=alpha)

# Initial conditions
r_init, n_init = geometry.random_initial_conditions()
# r_const = np.array([5,5,5],dtype=float)
# n_const = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],dtype=float)
# r_init, n_init = const_initial_conditions(config=config, r_const=r_const, n_const=n_const)

# Run simulation 
save_every = 1000
sim = Langevin_sim(config,I_fn=I_fn,f_fn=f_fn,r0=r_init, n0=n_init, geometry=geometry)
results = sim.run(save_every=save_every)
# sim.plot_trajectories()

r, n = results["r"], results["n"]
x, y, z = r[-1,0,:], r[-1,1,:], r[-1,2,:]
nx, ny, nz = n[-1,0,:], n[-1,1,:], n[-1,2,:]

# 2D currents xz

bins_x = bins_z = 10

Lx = config["Lx"]
Lz = config["Lz"]
dx = Lx/bins_x
dz = Lz/bins_z

ix = (x/dx).astype(int) # indices of x bins
iz = (z/dz).astype(int)

Jx = np.zeros((bins_x,bins_z))
Jz = np.zeros((bins_x, bins_z))
np.add.at(Jx, (ix, iz), nx)
np.add.at(Jz, (ix, iz), nz)
Jx /= dx
Jz /= dz

x_centers = (np.arange(bins_x) + 1/2)*dx
y_centers = (np.arange(bins_z) + 1/2)*dz
X, Z = np.meshgrid(x_centers, y_centers, indexing='ij')

plt.quiver(X, Z, Jx, Jz)
plt.show()


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