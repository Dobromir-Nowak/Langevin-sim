import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def make_gif(*data, plot_func, fps=20, title_func=None, show=True, **plot_kwargs):
    if not data:
        raise ValueError("make_gif requires at least one time-dependent data array")

    fig, ax = plt.subplots()

    nframes = len(data[0])
    for i, ar in enumerate(data[1:], start=1):
        if len(ar) != nframes:
            raise ValueError(
                f"all data arrays must have the same number of frames; "
                f"data[0] has {nframes}, data[{i}] has {len(ar)}"
            )

    def update(frame):

        ax.clear()
        
        frame_data = [ar[frame] for ar in data]
        plot_func(ax, *frame_data, **plot_kwargs)

        if title_func is None:
            ax.set_title(f"t = {frame}")
        else:
            ax.set_title(title_func(frame))

    ani = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=1000/fps   # interval measure in ms
    )
    if show:
        plt.show()
        print()
    return ani
