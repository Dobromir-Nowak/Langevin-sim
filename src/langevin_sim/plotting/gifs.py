import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Any
import numpy as np

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


def make_animation(
    *data: np.ndarray,
    plot_func: Callable[..., Any],
    fps: float = 20,
    frame_stride: int = 1,
    cumulative: bool = False,
    trail_length: int | None = None,
    title_func: Callable[[int], str] | None = None,
    show: bool = False,
    fig_kwargs: dict[str, Any] | None = None,
    **plot_kwargs: Any,
) -> FuncAnimation:
    """Create a Matplotlib animation from one or more time-dependent arrays.

    Parameters
    ----------
    data
        Arrays whose first axis is time. All arrays must contain the same
        number of saved states.
    plot_func
        Function called as ``plot_func(ax, *frame_data, **plot_kwargs)``.
    fps
        Playback frame rate. This sets the interactive preview interval; use
        the same value when saving unless a different playback rate is wanted.
    frame_stride
        Use every ``frame_stride``-th saved state.
    cumulative
        If False, pass ``array[frame]`` to ``plot_func``. If True, pass the
        history ending at the current frame, which is appropriate for drawing
        trajectories.
    trail_length
        With ``cumulative=True``, retain only this many saved states. ``None``
        draws the complete trajectory history.
    title_func
        Optional function receiving the original saved-state index.
    show
        Display the interactive Matplotlib window. In scripts that save the
        animation, keep this False and save before displaying.
    fig_kwargs
        Keyword arguments passed to ``plt.subplots``.
    plot_kwargs
        Keyword arguments forwarded to ``plot_func``.
    """
    if not data:
        raise ValueError("make_animation requires at least one time-dependent array")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be a positive integer")
    if trail_length is not None and trail_length <= 0:
        raise ValueError("trail_length must be positive or None")

    arrays = [np.asarray(array) for array in data]
    nframes = len(arrays[0])
    if nframes == 0:
        raise ValueError("time-dependent arrays must contain at least one frame")

    for i, array in enumerate(arrays[1:], start=1):
        if len(array) != nframes:
            raise ValueError(
                "all data arrays must have the same number of frames; "
                f"data[0] has {nframes}, data[{i}] has {len(array)}"
            )

    fig, ax = plt.subplots(**({} if fig_kwargs is None else fig_kwargs))
    frame_indices = np.arange(0, nframes, frame_stride, dtype=int)

    def update(frame: int):
        ax.clear()

        if cumulative:
            start = 0 if trail_length is None else max(0, frame + 1 - trail_length)
            frame_data = [array[start : frame + 1] for array in arrays]
        else:
            frame_data = [array[frame] for array in arrays]

        plot_func(ax, *frame_data, **plot_kwargs)

        if title_func is None:
            ax.set_title(f"frame = {frame}")
        else:
            ax.set_title(title_func(frame))

        return tuple(ax.get_children())

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / fps,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )

    if show:
        plt.show()

    return animation