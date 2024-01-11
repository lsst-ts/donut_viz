from functools import lru_cache

from astropy.table import Table, vstack
from matplotlib.patches import FancyArrowPatch
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
