import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

def psfPanel(
    xs,
    ys,
    psf,
    detname,
    fig=None,
    figsize=(11,14),
    cmap="cool",
    **kwargs
) -> Figure:
    """Make a per-detector psf scatter plot
    
    Subplots shows for each detector the psf retrieve from the zernike value
    for each pair of intra-extra focal images. The points are placed using
    pixel coordinates.

    Parameters
    ----------
    xs, ys: array of float, shape (ndet, npair)
        Points coordinates in pixel.
    psf: array of float, shape (ndet, npair)
        PSF value for each point.
    detname: list of strings, shape (ndet,)
        Detector labels.
    fig: matplotlib Figure, optional
        If provided, use this figure.  Default None.
    figsize: tuple of float, optional
        Figure size in inches.  Default (11, 12).
    cmap: str, optional
        Colormap name.  Default 'seismic'.
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
    gs = GridSpec(nrows=4, ncols=3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 0.1])
    axs = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
    ax_cbar = fig.add_subplot(gs[-1, :])

    # setting the detector size (maybe there is a more wise way to retrieve it from the data metadata)
    det_lim_y = (0., 4000.)
    det_lim_x = (0., 4072.)
    
    # setting the common colormap limits
    pmax = np.nanmax(psf)
    pmin = np.nanmin(psf)

    # cycling through the axes.
    for i, ax in enumerate(axs):
        im = ax.scatter(xs[i], ys[i], c=psf[i], cmap=cmap, vmax=pmax, vmin=pmin)
        ax.set_title(f"{detname[i]}: {np.nanmean(psf[i]):.3f} +/- {np.nanstd(psf[i]):.3f}")
        ax.set(xlim=det_lim_x, ylim=det_lim_y, xticks=[], yticks=[], aspect="equal")

    # setting the colorbar
    cb = fig.colorbar(im, cax=ax_cbar, location="bottom")
    cb.set_label(label="PSF width, arcsecond", fontsize="large")

    return fig