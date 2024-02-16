import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Axes


def _next_slice(ax: Axes):
    ax.index = (ax.index + 1) % ax.volume.shape[0]
    ax.images[0].set_array(ax.volume[ax.index])


def _previous_slice(ax: Axes):
    ax.index = (ax.index - 1) % ax.volume.shape[0]
    ax.images[0].set_array(ax.volume[ax.index])


def _process_key(event: KeyEvent):
    fig = event.canvas.figure
    ax = fig.axes[0]

    if event.key == "a":
        _previous_slice(ax)

    if event.key == "d":
        _next_slice(ax)

    fig.canvas.draw()


def visualize(volume: np.ndarray):
    """
    volume - will be of the type D, H, W
    """
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("key_press_event", _process_key)

    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], cmap="bone")

    plt.axis("off")
    plt.show()
