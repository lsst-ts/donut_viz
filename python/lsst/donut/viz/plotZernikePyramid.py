import galsim
import numpy as np
from pathlib import Path
import yaml
from utilities import get_cat, rose
from zernikePyramid import zernikePyramid


def plotZernikePyramid(
    butler,
    extra_exposure_id,
    intra_exposure_id=None,
    instrument='LSSTCam'
):
    cat, q, rot, rtp, band = get_cat(
        butler,
        extra_exposure_id,
        intra_exposure_id=intra_exposure_id,
        instrument=instrument
    )
    zs = cat['zs_OCS']

    fig = zernikePyramid(
        cat['thx_OCS'], cat['thy_OCS'],
        zs.T, cmap='seismic', s=2,
    )
    vecs_xy = {
        '$x_\mathrm{Opt}$':(1,0),
        '$y_\mathrm{Opt}$':(0,-1),
        '$x_\mathrm{Cam}$':(np.cos(rtp), -np.sin(rtp)),
        '$y_\mathrm{Cam}$':(-np.sin(rtp), -np.cos(rtp)),
    }
    rose(fig, vecs_xy, p0=(0.15, 0.8))

    vecs_NE = {
        'az':(1,0),
        'alt':(0,+1),
        'N':(np.sin(q), np.cos(q)),
        'E':(np.sin(q-np.pi/2), np.cos(q-np.pi/2))
    }
    rose(fig, vecs_NE, p0=(0.85, 0.8))

    filename = f'zernikePyramid_{extra_exposure_id}.png'
    fig.savefig(filename)

    # We want residuals from the intrinsic design too.
    path = Path(__file__).parent.parent.parent.parent.parent / 'data'
    path /= f'intrinsic_dz_{band}.yaml'
    coefs = np.array(yaml.safe_load(open(path, 'r')))
    dzs = galsim.zernike.DoubleZernike(
        coefs,
        uv_outer=np.deg2rad(1.82),
        xy_outer=4.18,
        xy_inner=4.18*0.612,
    )
    intrinsic = np.array([z.coef for z in dzs(cat['thx_OCS'], cat['thy_OCS'])])
    resid = zs - intrinsic[:, 4:23]

    fig2 = zernikePyramid(
        cat['thx_OCS'], cat['thy_OCS'],
        resid.T, cmap='seismic', s=2,
    )
    rose(fig2, vecs_xy, p0=(0.15, 0.8))
    rose(fig2, vecs_NE, p0=(0.85, 0.8))
    filename2 = f'zernikePyramidResid_{extra_exposure_id}.png'
    fig2.savefig(filename2)

    # Plot the actual intrinsic too.
    fig3 = zernikePyramid(
        cat['thx_OCS'], cat['thy_OCS'],
        intrinsic[:, 4:23].T, cmap='seismic', s=2,
    )
    rose(fig3, vecs_xy, p0=(0.15, 0.8))
    rose(fig3, vecs_NE, p0=(0.85, 0.8))
    filename3 = f'zernikePyramidIntrinsic_{extra_exposure_id}.png'
    fig3.savefig(filename3)



if __name__ == "__main__":
    from lsst.daf.butler import Butler
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('butler', type=str)
    parser.add_argument('collection', type=str)
    parser.add_argument('extra_exposure_id', type=int)
    parser.add_argument('--intra_exposure_id', type=int, default=None)
    parser.add_argument('--instrument', type=str, default='LSSTCam')
    args = parser.parse_args()

    butler = Butler(args.butler, collections=args.collection)
    plotZernikePyramid(
        butler,
        args.extra_exposure_id,
        intra_exposure_id=args.intra_exposure_id,
        instrument=args.instrument
    )
