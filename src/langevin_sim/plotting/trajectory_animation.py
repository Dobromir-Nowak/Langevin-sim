import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

def make_trajectory_animation(
    r,
    config,
    excluded_axis=0,
    n_show=100,
    frame_stride=1,
    trail_length=100,
    fps=20,
    title_func=None,
):
    """
    r shape: (T, 3, N)
    """

    side_names = ["Lx", "Ly", "Lz"]
    axis_names = ["x", "y", "z"]

    plot_axes = [i for i in range(3) if i != excluded_axis]
    axis_0, axis_1 = plot_axes

    n_show = min(n_show, r.shape[2])
    particle_ids = np.linspace(
        0,
        r.shape[2] - 1,
        n_show,
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for spine in ax.spines.values():
        spine.set_visible(False)

    rect = Rectangle(
        (0, 0),
        config[side_names[axis_0]],
        config[side_names[axis_1]],
        fill=False,
        ec="0.4",
        alpha=0.5,
        lw=1,
    )
    ax.add_patch(rect)

    pad = 0.02 * max(config[side_names[axis_0]], config[side_names[axis_1]])
    ax.set_xlim(-pad, config[side_names[axis_0]] + pad)
    ax.set_ylim(-pad, config[side_names[axis_1]] + pad)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(axis_names[axis_0])
    ax.set_ylabel(axis_names[axis_1])

    lines = [
        ax.plot([], [], lw=1, color="green")[0]
        for _ in particle_ids
    ]

    frame_indices = np.arange(
        0,
        r.shape[0],
        frame_stride,
        dtype=int,
    )

    def update(frame):
        start = max(0, frame + 1 - trail_length)

        for line, particle_id in zip(lines, particle_ids):
            x = r[start:frame + 1, axis_0, particle_id]
            y = r[start:frame + 1, axis_1, particle_id]

            line.set_data(x, y)

        if title_func is None:
            ax.set_title(f"Frame {frame}")
        else:
            ax.set_title(title_func(frame))

        return lines

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps,
        blit=True,
        repeat=False,
        cache_frame_data=False,
    )

    return animation