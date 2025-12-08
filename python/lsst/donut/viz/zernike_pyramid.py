from typing import Any, Callable

import galsim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


# From https://joseph-long.com/writing/colorbars/
def colorbar(mappable: plt.cm.ScalarMappable) -> Colorbar:
    """Add a colorbar to a mappable plot element.

    Parameters
    ----------
    mappable : matplotlib mappable
        The mappable plot element (e.g., scatter, imshow, etc).

    Returns
    -------
    cbar : matplotlib Colorbar
        The colorbar.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def zernikePyramid(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    noll_indices: np.ndarray,
    figsize: tuple = (13, 8),
    vmin: float = -1,
    vmax: float = 1,
    vdim: bool = True,
    s: float = 5,
    title: str | None = None,
    callback: Callable | None = None,
    filename: str | None = None,
    fig: Figure | None = None,
    cmap: str = "seismic",
    **kwargs: Any,
) -> Figure:
    """Make a multi-zernike plot in a pyramid shape.

    Subplots show individual Zernikes over a range of x and y (presumably a
    field of view).

    Parameters
    ----------
    xs, ys: array of float
        Field angles (or other spatial coordinate over which to plot Zernikes)
    zs: array of float, shape (jmax, xymax)
        Zernike values.  First index labels the particular Zernike coefficient,
        second index labels spatial coordinate.  First index implicitly starts
        at j=jmin (defocus by default).
    noll_indices: np.ndarray
        Noll indices for zernikes in zs.
    figsize: tuple of float, optional
        Figure size in inches.  Default (13, 8).
    vmin, vmax: float, optional
        Color scale limits.  Default (-1, 1).
    vdim: bool, optional
        If True, scale vmin and vmax by the Zernike radial order.
        Default True.
    s: float, optional
        Marker size.  Default 5.
    callback: callable, optional
        A callable to execute just before adjusting axis locations.  Useful for
        setting suptitle, for example.  Takes two keyword arguments, fig for
        the Figure, and axes for a zernike-indexed dict of plot Axes.
        Default: None
    filename: str, optional
        If provided, save figure to this filename.  Default None.
    fig: matplotlib Figure, optional
        If provided, use this figure.  Default None.
    cmap: str, optional
        Colormap name.  Default 'seismic'.
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: matplotlib Figure
        The figure.
    """
    jmin = min(noll_indices)
    jmax = max(noll_indices)
    jdict = {x: y for x, y in zip(noll_indices, range(len(noll_indices)))}
    nmax, _ = galsim.zernike.noll_to_zern(jmax)
    nmin, _ = galsim.zernike.noll_to_zern(jmin)

    nrow = nmax - nmin + 1
    ncol = nmax + 1
    gridspec = GridSpec(nrow, ncol)

    def shift(pos: Rectangle, amt: float) -> list[float]:
        return [pos.x0 + amt, pos.y0, pos.width, pos.height]

    def shiftAxes(axes: list[Axes], amt: float) -> None:
        for ax in axes:
            ax.set_position(shift(ax.get_position(), amt))

    if fig is None:
        fig = Figure(figsize=figsize, **kwargs)
    axes = {}
    shiftLeft = []
    shiftRight = []
    for j in noll_indices:
        n, m = galsim.zernike.noll_to_zern(j)
        if n % 2 == 0:
            row, col = n - nmin, m // 2 + ncol // 2
        else:
            row, col = n - nmin, (m - 1) // 2 + ncol // 2
        subplotspec = gridspec.new_subplotspec((row, col))
        axes[j] = fig.add_subplot(subplotspec)
        axes[j].set_aspect("equal")
        if nmax % 2 == 0 and (nmax - n) % 2 == 1:
            shiftRight.append(axes[j])
        if nmax % 2 == 1 and (nmax - n) % 2 == 1:
            shiftLeft.append(axes[j])

    cbar = {}
    for j, ax in axes.items():
        if j not in noll_indices:
            continue
        n, _ = galsim.zernike.noll_to_zern(j)
        ax.set_title("Z{}".format(j))
        if vdim:
            _vmin = vmin / n
            _vmax = vmax / n
        else:
            _vmin = vmin
            _vmax = vmax
        scat = ax.scatter(
            xs,
            ys,
            c=zs[jdict[j]],
            s=s,
            linewidths=0.5,
            cmap=cmap,
            rasterized=True,
            vmin=_vmin,
            vmax=_vmax,
        )
        cbar[j] = colorbar(scat)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    if title is not None:
        raise DeprecationWarning("title argument is deprecated.  Use a callback.")
        fig.suptitle(title, x=0.1)

    if callback is not None:
        callback(fig=fig, axes=axes)

    fig.tight_layout()
    # Assume we always have Z4 and Z5?
    amt = 0.5 * (axes[4].get_position().x0 - axes[5].get_position().x0)
    shiftAxes(shiftLeft, -amt)
    shiftAxes(shiftRight, amt)

    if filename:
        fig.savefig(filename)

    return fig
