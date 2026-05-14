import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def make_gif(*data, plot_func, fps=20):

    fig, ax = plt.subplots()

    nframes = len(data[0])

    def update(frame):

        ax.clear()

        frame_data = [ar[frame] for ar in data]
        plot_func(ax, *frame_data)

        ax.set_title(f"t = {frame}")

    ani = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=1000/fps   # interval measure in ms
    )
    plt.show()
    return ani



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
datax = data[:,:,0]
make_gif(datax, plot_func=plot_hist_ax)

make_gif(data, plot_func=plot_density)