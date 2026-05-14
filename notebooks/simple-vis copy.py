import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def make_gif(data, plot_func, filename="animation.gif", fps=20):

    fig, ax = plt.subplots()

    def update(frame):

        ax.clear()

        plot_func(ax, data[frame])

        ax.set_title(f"t = {frame}")

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data),
        interval=1000/fps   # interval measure in ms
    )
    ani.save(filename, writer="pillow", fps=fps)
    plt.show()
    # plt.close(fig)


Nt = 40
Nx = Ny = 50

data = np.random.randn(Nt, Nx, Ny)

def plot_density(ax, state):

    ax.imshow(
        state,
        origin="lower",
        cmap="viridis"
    )

from langevin_sim.plotting.plots_ax import plot_hist_ax

# datax = data[:,:,0]
# make_gif(datax, plot_func=plot_hist_ax)
make_gif(data, plot_func=plot_density)