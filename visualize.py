import argparse

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pyvista as pv
from matplotlib.backend_bases import KeyEvent
from matplotlib.figure import Axes, Figure
from matplotlib.widgets import Slider


def _load_nibale_file(path: str) -> np.ndarray:
    epi_img = nibabel.load(path)
    data = epi_img.get_fdata()
    return data


def _vis_3d(data: np.ndarray):
    vol = pv.wrap(data)
    plotter = pv.Plotter()
    plotter.add_volume(vol)
    plotter.show()


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


def _update_scroll(idx: int, fig: Figure):
    ax = fig.axes[0]
    vol = ax.volume[idx]
    ax.images[0].set_array(vol)
    print(vol.shape)
    # print(np.unique(vol, return_counts=True)[1])
    fig.canvas.draw()


def _visualize(volume: np.ndarray):
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("key_press_event", _process_key)

    slider_ax = fig.add_axes([0.2, 0.02, 0.65, 0.01])
    slider = Slider(
        ax=slider_ax,
        valmin=0,
        valmax=volume.T.shape[0] - 1,
        valstep=1,
        valinit=volume.T.shape[0] // 2,
        label="Current slice",
    )

    slider.on_changed(lambda x: _update_scroll(x, fig))

    ax.volume = volume.T
    ax.index = ax.volume.shape[0] // 2
    ax.imshow(ax.volume[ax.index])

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("type", choices=["slice", "3d"])

    args = parser.parse_args()
    data = _load_nibale_file(args.filepath)

    if args.type == "slice":
        _visualize(data)

    if args.type == "3d":
        _vis_3d(data)
