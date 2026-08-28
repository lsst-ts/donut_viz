# This file is part of donut-viz.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path

import galsim
import numpy as np
import yaml
from utilities import get_cat, rose
from zernike_pyramid import zernikePyramid

from lsst.daf.butler import Butler


def plotZernikePyramid(
    butler: Butler, extra_exposure_id: int, intra_exposure_id: int | None = None, instrument: str = "LSSTCam"
) -> None:
    cat, q, rot, rtp, band = get_cat(
        butler,
        extra_exposure_id,
        intra_exposure_id=intra_exposure_id,
        instrument=instrument,
    )
    zs = cat["zs_OCS"]
    noll_indices = cat.meta["nollIndices"]

    fig = zernikePyramid(
        cat["thx_OCS"],
        cat["thy_OCS"],
        zs.T,
        noll_indices,
        cmap="seismic",
        s=2,
    )
    vecs_xy = {
        r"$x_\mathrm{Opt}$": (1, 0),
        r"$y_\mathrm{Opt}$": (0, -1),
        r"$x_\mathrm{Cam}$": (np.cos(rtp), -np.sin(rtp)),
        r"$y_\mathrm{Cam}$": (-np.sin(rtp), -np.cos(rtp)),
    }
    rose(fig, vecs_xy, p0=(0.15, 0.8))

    vecs_NE = {
        "az": (1, 0),
        "alt": (0, +1),
        "N": (np.sin(q), np.cos(q)),
        "E": (np.sin(q - np.pi / 2), np.cos(q - np.pi / 2)),
    }
    rose(fig, vecs_NE, p0=(0.85, 0.8))

    filename = f"zernikePyramid_{extra_exposure_id}.png"
    fig.savefig(filename)

    # We want residuals from the intrinsic design too.
    path = Path(__file__).parent.parent.parent.parent.parent / "data"
    path /= f"intrinsic_dz_{band}.yaml"
    coefs = np.array(yaml.safe_load(open(path, "r")))
    dzs = galsim.zernike.DoubleZernike(
        coefs,
        uv_outer=np.deg2rad(1.82),
        xy_outer=4.18,
        xy_inner=4.18 * 0.612,
    )
    intrinsic = np.array([z.coef for z in dzs(cat["thx_OCS"], cat["thy_OCS"])])
    resid = zs - intrinsic[:, 4:29]

    fig2 = zernikePyramid(
        cat["thx_OCS"],
        cat["thy_OCS"],
        resid.T,
        noll_indices,
        cmap="seismic",
        s=2,
    )
    rose(fig2, vecs_xy, p0=(0.15, 0.8))
    rose(fig2, vecs_NE, p0=(0.85, 0.8))
    filename2 = f"zernikePyramidResid_{extra_exposure_id}.png"
    fig2.savefig(filename2)

    # Plot the actual intrinsic too.
    fig3 = zernikePyramid(
        cat["thx_OCS"],
        cat["thy_OCS"],
        intrinsic[:, noll_indices].T,
        noll_indices,
        cmap="seismic",
        s=2,
    )
    rose(fig3, vecs_xy, p0=(0.15, 0.8))
    rose(fig3, vecs_NE, p0=(0.85, 0.8))
    filename3 = f"zernikePyramidIntrinsic_{extra_exposure_id}.png"
    fig3.savefig(filename3)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("butler", type=str)
    parser.add_argument("collection", type=str)
    parser.add_argument("extra_exposure_id", type=int)
    parser.add_argument("--intra_exposure_id", type=int, default=None)
    parser.add_argument("--instrument", type=str, default="LSSTCam")
    args = parser.parse_args()

    butler = Butler.from_config(args.butler, collections=args.collection)
    plotZernikePyramid(
        butler,
        args.extra_exposure_id,
        intra_exposure_id=args.intra_exposure_id,
        instrument=args.instrument,
    )
