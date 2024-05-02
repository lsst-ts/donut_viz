from functools import lru_cache

from astropy.table import Table, vstack
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator
from tqdm import trange
import galsim
import numpy as np

from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.geom import Point2D


@lru_cache()
def get_cat(butler, extra_exposure_id, intra_exposure_id=None, instrument='LSSTCam'):
    if intra_exposure_id is None:
        intra_exposure_id = extra_exposure_id+1
    camera = butler.get("camera", instrument=instrument)
    band = butler.registry.expandDataId(visit=extra_exposure_id)["band"]
    refdet = camera['R22_S11'].getId()  # Works for both LsstCam and LsstComCam
    cats = []
    for detnum in trange(189):
        det = camera[detnum]
        # if det.getName().startswith("R01"):
        #     continue
        tform = det.getTransform(PIXELS, FIELD_ANGLE)
        cat = butler.get("donutCatalog", visit=extra_exposure_id, detector=detnum)
        cat2 = butler.get("donutCatalog", visit=intra_exposure_id, detector=detnum)
        if len(cat) != len(cat2):
            continue
        cat = Table.from_pandas(cat)  # kill the pandas!

        pts = tform.applyForward([Point2D(x, y) for x, y in zip(cat['centroid_x'], cat['centroid_y'])])
        cat['thx_CCS'] = [pt.y for pt in pts]  # Note x,y => y,x
        cat['thy_CCS'] = [-pt.x for pt in pts]  # Why the - sign?

        zs = butler.get("zernikeEstimateRaw", visit=extra_exposure_id, detector=detnum)
        cat['zs_CCS'] = zs
        cats.append(cat)
    cat = vstack(cats)
    visitInfo = butler.get("postISRCCD.visitInfo", exposure=extra_exposure_id, detector=refdet)
    q = visitInfo.boresightParAngle.asRadians()
    rot = visitInfo.boresightRotAngle.asRadians()
    rtp = q-rot-np.pi/2
    cat['thx_OCS'] = np.cos(rtp)*cat['thx_CCS'] + np.sin(rtp)*cat['thy_CCS']
    cat['thy_OCS'] = -np.sin(rtp)*cat['thx_CCS'] + np.cos(rtp)*cat['thy_CCS']
    cat['th_N'] = np.cos(q)*cat['thx_CCS'] + np.sin(q)*cat['thy_CCS']
    cat['th_E'] = -np.sin(q)*cat['thx_CCS'] + np.cos(q)*cat['thy_CCS']
    rotM = galsim.zernike.zernikeRotMatrix(22, rtp)[4:,4:]
    cat['zs_OCS'] = cat['zs_CCS'] @ rotM
    return cat, q, rot, rtp, band


def rose(fig, vecs, p0=(0.105, 0.105), length=0.05):
    size = fig.get_size_inches()
    ratio = size[0]/size[1]

    for k in vecs.keys():
        if k in ['N', 'E', 'W', 'S']:
            color='r'
        elif k in ['alt', 'az']:
            color='g'
        else:
            color='k'
        dp = length * vecs[k][0], length*ratio * vecs[k][1]
        p1 = p1 = p0[0]+dp[0], p0[1]+dp[1]

        fig.patches.append(
            FancyArrowPatch(
                p0, p1,
                transform=fig.transFigure,
                color=color,
                arrowstyle='-|>',
                mutation_scale=10, lw=1.5
            )
        )

        dp = 1.2 * length * vecs[k][0], 1.2 * length*ratio * vecs[k][1]
        p1 = p1 = p0[0]+dp[0], p0[1]+dp[1]
        fig.text(p1[0], p1[1], k, color=color, ha='center', va='center')


def add_rotated_axis(fig, xy, wh, th):
    """
    Parameters
    ----------

    fig : matplotlib figure
        The figure to add the rotated axis to.
    xy : (float, float)
        The center of the rotated axis.
    wh : (float, float)
        The width and height of the rotated axis.
    th : float
        The angle of the rotated axis in degrees.

    Returns
    -------
    ax : matplotlib axes
        The rotated axis.
    aux_ax : matplotlib axes
        The original axis.
    """
    x, y = xy
    w, h = wh
    s1 = w * np.abs(np.cos(np.deg2rad(th))) + h * np.abs(np.sin(np.deg2rad(th)))
    s2 = w * np.abs(np.sin(np.deg2rad(th))) + h * np.abs(np.cos(np.deg2rad(th)))
    tr = Affine2D().rotate_deg(th)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr,
        extremes=(0, w, 0, h),
        grid_locator1=FixedLocator([]),
        grid_locator2=FixedLocator([]),
    )

    ax = fig.add_axes(
        [x-s1/2, y-s2/2, s1, s2],
        axes_class=floating_axes.FloatingAxes,
        grid_helper=grid_helper
    )
    aux_ax = ax.get_aux_axes(tr)

    return ax, aux_ax

def get_instrument_channel_name(instrument):
    """Get the instrument channel name for the current instrument.

    This is the RubinTV channel required to upload.

    Parameters
    ----------
    instrument : `str`
        The instrument name, e.g. 'LSSTCam'.

    Returns
    -------
    channel : `str`
        The channel prefix name.
    """
    match instrument:
        case 'LSSTCam':
            return 'lsstcam_aos'
        case 'LSSTCamSim':
            return 'lsstcam_sim_aos'
        case 'LSSTComCam':
            return 'comcam_aos'
        case 'LSSTComCamSim':
            return 'comcam_sim_aos'
        case _:
            raise ValueError(f'Unknown instrument {instrument}')

def get_day_obs_seq_num_from_visitid(visit):
    """Get the dayObs and seqNum from a visit ID.

    Parameters
    ----------
    visit : `int`
        The visit ID.

    Returns
    -------
    day_obs : `int`
        The day_obs.
    seq_num : `int`
        The seq_num.
    """

    day_obs = visit // 100_000 % 1_000_000 + 20_000_000
    seq_num = visit % 100_000

    return day_obs, seq_num
