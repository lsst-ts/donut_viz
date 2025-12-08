from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


def psfPanel(
    xs: list[list[float]],
    ys: list[list[float]],
    psf: list[list[float]],
    detname: list[str],
    dettype: str = "LSSTComCam",
    fig: Figure | None = None,
    figsize: tuple[float, float] = (11, 14),
    maxcol: int = 3,
    cmap: str = "cool",
    **kwargs: Any,
) -> Figure:
    """Make a per-detector psf scatter plot

    Subplots shows for each detector the psf retrieved from the zernike value
    for each pair of intra-extra focal images. The points are placed using
    pixel coordinates.

    Parameters
    ----------
    xs, ys: list of list[float], shape (ndet, npair)
        Points coordinates in pixel.
    psf: list of list[float], shape (ndet, npair)
        PSF value for each point.
    detname: list of strings, shape (ndet,)
        Detector labels.
    fig: matplotlib Figure, optional
        If provided, use this figure.  Default None.
    figsize: tuple of float, optional
        Figure size in inches.  Default (11, 12).
    cmap: str, optional
        Colormap name.  Default 'cool'.
    maxcol: int, optional
        Maximum number of columns to use while creating the subplots grid.
        Default 3
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: matplotlib Figure
        The figure.
    """

    # generating figure if None
    if fig is None:
        fig = Figure(figsize=figsize, **kwargs)

    # creating the gridspec grid (3x3 equal axes and the bottom cbar ax)
    if len(detname) < maxcol:
        det_nrows = 1
        ncols = len(detname)
    else:
        det_nrows = len(detname) // maxcol + 1 if len(detname) % maxcol != 0 else len(detname) // maxcol
        ncols = maxcol

    gs = GridSpec(
        nrows=det_nrows + 1,  # add a final row for the cbar
        ncols=ncols,
        figure=fig,
        width_ratios=[1.0] * ncols,
        height_ratios=[1.0] * det_nrows + [0.1],
    )
    axs = []
    for i in range(len(detname)):
        axs.append(fig.add_subplot(gs[i // ncols, i % ncols]))
    ax_cbar = fig.add_subplot(gs[-1, :])

    # setting the detector size
    # (maybe there is a more wise way to retrieve it from the data metadata)
    match dettype:
        case "LSSTComCam":
            det_lim_y = (0.0, 4000.0)
            det_lim_x = (0.0, 4072.0)
        case "LSSTCam":
            det_lim_y = (0.0, 2000.0)
            det_lim_x = (0.0, 4072.0)
        case _:
            raise ValueError("Detector type not known")

    # setting the common colormap limits
    pmax = np.nanmax(np.concatenate(psf))
    pmin = np.nanmin(np.concatenate(psf))

    # cycling through the axes.
    for i, dn in enumerate(detname):
        axs[i].set(xlim=det_lim_x, ylim=det_lim_y, xticks=[], yticks=[], aspect="equal")
        if len(psf[i]) == 0:
            continue
        im = axs[i].scatter(xs[i], ys[i], c=psf[i], cmap=cmap, vmax=pmax, vmin=pmin)
        axs[i].set_title(f"{dn}: {np.nanmean(psf[i]):.3f} +/- {np.nanstd(psf[i]):.3f}")

    # setting the colorbar
    cb = fig.colorbar(im, cax=ax_cbar, location="bottom")
    cb.set_label(label="PSF width, arcsecond", fontsize="large")

    return fig
