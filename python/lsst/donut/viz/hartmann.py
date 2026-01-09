from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from random import random
from time import sleep
from typing import Any

import astropy.units as units
import batoid
import danish
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as ct
import numpy as np
import requests
import yaml
from astropy.coordinates import Angle
from astropy.table import QTable, vstack
from astropy.time import Time
from batoid_rubin import LSSTBuilder
from galsim.zernike import zernikeBasis, zernikeGradBases, zernikeRotMatrix
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.ip.isr import IsrTaskLSST
from lsst.meas.algorithms import Stamp, Stamps, SubtractBackgroundTask
from lsst.pex.exceptions import LengthError
from lsst.ts.ofc import BendModeToForce, OFCData
from matplotlib.figure import Figure
from scipy.optimize import least_squares
from scipy.signal import correlate
from skimage.feature import peak_local_max

__all__ = [
    "HartmannSensitivityAnalysisConfig",
    "HartmannSensitivityAnalysisConnections",
    "HartmannSensitivityAnalysis",
]


class EFD:
    """Class to query USDF EFD via REST API."""

    def __init__(self, site: str = "usdf_efd"):
        creds_service = f"https://roundtable.lsst.codes/segwarides/creds/{site}"
        attempt = 0
        while attempt < 5:
            attempt += 1
            try:
                efd_creds = requests.get(creds_service, timeout=10).json()
            except (requests.RequestException, ValueError, KeyError):
                sleep(0.5 + random())
                continue
            else:
                break
        else:
            raise ConnectionError("Unable to retrieve credentials after 5 attempts")

        self.auth = efd_creds["username"], efd_creds["password"]
        self.url = "https://usdf-rsp.slac.stanford.edu/influxdb-enterprise-data/query"

    def get_most_recent_row_before(self, topic, time, n_retries=10, where=None):
        # Stealing from summit_utils and rubin_nights
        earliest = Time("2019-01-01")
        if time < earliest:
            raise ValueError(f"Time {time} is before EFD start time.")
        earliest = max(earliest, time - 240 * units.min)
        query = f'select * from "{topic}" '
        query += f"where time > '{earliest.utc.isot}Z' and time <= '{time.utc.isot}Z'"
        params = {"db": "efd", "q": query}

        attempt = 0
        while attempt < n_retries:
            attempt += 1
            try:
                response = requests.get(self.url, auth=self.auth, params=params).json()
            except ValueError:
                sleep(0.5 + random())
                continue
            statement = response["results"][0]
            if "series" not in statement:
                # Also possibly transient...
                sleep(0.5 + random())
                continue
            # Success
            series = statement["series"][0]
            result = QTable(rows=series.get("values", []), names=series["columns"])
            if where is not None and len(result) > 0:
                result = result[where(result)]
            if "time" in result.colnames:
                result["time"] = Time(result["time"], format="isot", scale="utc")
            return result[-1]
        raise ValueError(
            f"No data found for topic {topic} at time {time} after {n_retries} retries."
        )

    def get_efd_data(
        self,
        topic,
        begin,
        end,
        n_retries=10,
    ):
        query = f'select * from "{topic}" '
        query += f"where time > '{begin.utc.isot}Z' and time <= '{end.utc.isot}Z'"
        params = {"db": "efd", "q": query}

        attempt = 0
        while attempt < n_retries:
            attempt += 1
            try:
                response = requests.get(self.url, auth=self.auth, params=params).json()
            except ValueError:
                sleep(0.5 + random())
                continue
            statement = response["results"][0]
            if "series" not in statement:
                # Also possibly transient...
                sleep(0.5 + random())
                continue
            # Success
            series = statement["series"][0]

            result = QTable(rows=series.get("values", []), names=series["columns"])
            if "time" in result.colnames:
                result["time"] = Time(result["time"], format="isot", scale="utc")
            return result
        raise ValueError(
            f"No data found for topic {topic} between {begin} and {end} after {n_retries} retries."
        )


def get_rtp(exposure):
    visit_info = exposure.info.getVisitInfo()
    rsp = visit_info.boresightRotAngle.asDegrees() * units.deg
    q = visit_info.boresightParAngle.asDegrees() * units.deg
    rtp = q - rsp - 90.0 * units.deg
    return rtp


def ccs_to_ocs(x_ccs, y_ccs, rtp: Angle):
    crtp, srtp = np.cos(rtp).value, np.sin(rtp).value
    x_ocs = x_ccs * crtp - y_ccs * srtp
    y_ocs = x_ccs * srtp + y_ccs * crtp
    return x_ocs, y_ocs


def ocs_to_ccs(x_ocs, y_ocs, rtp: Angle):
    return ccs_to_ocs(x_ocs, y_ocs, -rtp)


def ccs_to_dvcs(x_ccs, y_ccs):
    # Works for field angle and focal plane.  May need more care for pixels
    # due to corner WF sensor rotations.
    return y_ccs, x_ccs


def dvcs_to_ccs(x_dvcs, y_dvcs):
    return y_dvcs, x_dvcs


def get_rays(telescope, u_ocs, v_ocs, x_ocs, y_ocs):
    rays = batoid.RayVector.fromStop(
        x=u_ocs,
        y=v_ocs,
        optic=telescope,
        wavelength=700e-9,
        theta_x=x_ocs,
        theta_y=y_ocs,
    )
    return rays


def trace_ocs_to_ccd(rays, telescope, det, rtp: Angle):
    fp = telescope.trace(rays.copy())  # OCS FOCAL_PLANE
    x_fp_ocs = fp.x * 1e3  # m to mm
    y_fp_ocs = fp.y * 1e3  # m to mm
    x_fp_ccs, y_fp_ccs = ocs_to_ccs(x_fp_ocs, y_fp_ocs, rtp)
    x_fp_dvcs, y_fp_dvcs = ccs_to_dvcs(x_fp_ccs, y_fp_ccs)
    x_ccd_dvcs, y_ccd_dvcs = [], []
    for x, y in zip(x_fp_dvcs, y_fp_dvcs):
        x_ccd, y_ccd = det.transform(Point2D(x, y), FOCAL_PLANE, PIXELS)
        x_ccd_dvcs.append(x_ccd)
        y_ccd_dvcs.append(y_ccd)
    x_ccd_dvcs = np.array(x_ccd_dvcs)
    y_ccd_dvcs = np.array(y_ccd_dvcs)
    return x_ccd_dvcs, y_ccd_dvcs


def pupil_to_pixel(u_ocs, v_ocs, dx, dy, zTA_ccs, rtp, nrot):
    u_ccs, v_ccs = ocs_to_ccs(u_ocs, v_ocs, rtp)
    x, y = danish.pupil_to_focal(
        u_ccs,
        v_ccs,
        aberrations=zTA_ccs,
        R_outer=4.18,
        R_inner=4.18 * 0.612,
        focal_length=10.31,
    )
    # CCS to DVCS
    if nrot % 4 == 1:
        x, y = -y, x
    elif nrot % 4 == 2:
        x, y = -x, -y
    elif nrot % 4 == 3:
        x, y = y, -x
    x, y = y, x

    x /= 10e-6  # m to pixels
    y /= 10e-6
    x += dx
    y += dy

    return x, y


def pixel_to_pupil(x, y, dx, dy, zTA_ccs, rtp, nrot):
    x -= dx
    y -= dy
    x *= 10e-6  # pixels to m
    y *= 10e-6
    x, y = y, x  # DVCS to CCS
    if nrot % 4 == 1:
        x, y = y, -x
    elif nrot % 4 == 2:
        x, y = -x, -y
    elif nrot % 4 == 3:
        x, y = -y, x
    u_ccs, v_ccs = danish.focal_to_pupil(
        x,
        y,
        aberrations=zTA_ccs,
        R_outer=4.18,
        R_inner=4.18 * 0.612,
        focal_length=10.31,
    )
    u_ocs, v_ocs = ocs_to_ccs(u_ccs, v_ccs, rtp)
    return u_ocs, v_ocs


def fit_danish(telescope, x_ccs, y_ccs, stamp, nrot, verbose=None):
    if verbose is None:
        verbose = 0
    with open(Path(danish.datadir) / "RubinObsc.yaml") as f:
        mask_params = yaml.safe_load(f)

    zTA = (
        batoid.zernikeTA(
            telescope,
            theta_x=x_ccs,
            theta_y=y_ccs,
            wavelength=700e-9,
            nrad=20,
            jmax=78,
            eps=0.612,
            focal_length=10.31,
        )
        * 700e-9
    )

    factory = danish.DonutFactory(
        R_outer=4.18,
        R_inner=4.18 * 0.612,
        mask_params=mask_params,
        pixel_scale=10e-6,
    )

    arr = np.rot90(stamp.stamp_im.image.array.T, nrot)  # DVCS -> CCS

    fitter = danish.SingleDonutModel(
        factory, z_ref=zTA, thx=x_ccs, thy=y_ccs, z_terms=(), npix=arr.shape[0]
    )
    guess = [0.0, 0.0, 0.8]

    fit = least_squares(
        fitter.chi,
        guess,
        jac=fitter.jac,
        bounds=[(-10, -10, 0.5), (10, 10, 2.5)],
        x_scale=[1.0, 1.0, 0.01],
        args=(arr, 1000),
        verbose=verbose,
    )
    dx, dy, fwhm, *z_fit = fit.x
    model = fitter.model(dx, dy, fwhm, z_fit)
    # Convert offsets to actual pixel coordinates
    dx_pix = dx / (3600 * np.rad2deg(1 / 10.31) * 10e-6)  # arcsec -> pix
    dy_pix = dy / (3600 * np.rad2deg(1 / 10.31) * 10e-6)

    # CCS -> DVCS
    model = np.rot90(model, -nrot).T
    # CCS -> DVCS
    if nrot % 4 == 1:
        dx_pix, dy_pix = -dy_pix, dx_pix
    elif nrot % 4 == 2:
        dx_pix, dy_pix = -dx_pix, -dy_pix
    elif nrot % 4 == 3:
        dx_pix, dy_pix = dy_pix, -dx_pix
    dx_pix, dy_pix = dy_pix, dx_pix

    # Warning: Might be off if not square
    dx_pix += arr.shape[1] // 2
    dy_pix += arr.shape[0] // 2
    return dx_pix, dy_pix, model, zTA


@lru_cache
def get_m2_bmf():
    """Get the M2 BendModeToForce object"""
    return BendModeToForce("M2", OFCData("lsst"))


@lru_cache
def get_m1m3_bmf():
    """Get the M1M3 BendModeToForce object"""
    return BendModeToForce("M1M3", OFCData("lsst"))


def get_state(exposure, efd_client):
    visitInfo = exposure.info.getVisitInfo()
    begin = visitInfo.date.toAstropy()
    begin -= visitInfo.exposureTime * units.s / 2
    end = begin + visitInfo.exposureTime * units.s

    out = np.zeros(50, dtype=np.float64)

    m2_table = efd_client.get_most_recent_row_before(
        "lsst.sal.MTHexapod.logevent_uncompensatedPosition",
        time=end,
        where=lambda table: table["salIndex"] == 2,
    )
    out[0] = m2_table["z"]
    out[1] = m2_table["x"]
    out[2] = m2_table["y"]
    out[3] = m2_table["u"]
    out[4] = m2_table["v"]

    cam_table = efd_client.get_most_recent_row_before(
        "lsst.sal.MTHexapod.logevent_uncompensatedPosition",
        time=end,
        where=lambda table: table["salIndex"] == 1,
    )
    out[5] = cam_table["z"]
    out[6] = cam_table["x"]
    out[7] = cam_table["y"]
    out[8] = cam_table["u"]
    out[9] = cam_table["v"]
    m1m3_event = efd_client.get_most_recent_row_before(
        "lsst.sal.MTM1M3.logevent_appliedActiveOpticForces",
        time=end,
    )
    m1m3_forces = np.empty((156,))
    for i in range(156):
        m1m3_forces[i] = m1m3_event[f"zForces{i}"]
    out[10:30] = get_m1m3_bmf().bending_mode(m1m3_forces)

    m2_telemetry = efd_client.get_efd_data(
        "lsst.sal.MTM2.axialForce",
        begin=begin,
        end=end,
    )
    nrow = len(m2_telemetry)
    m2_forces = np.empty((nrow, 72), dtype=np.float64)
    for i in range(72):
        m2_forces[:, i] = m2_telemetry[f"applied{i}"]
    m2_forces = np.mean(m2_forces, axis=0)
    out[30:] = get_m2_bmf().bending_mode(m2_forces)

    return out


def get_parabola_vertex(v_m1, v_0, v_p1):
    denom = 2 * (v_m1 - 2 * v_0 + v_p1)
    if denom == 0:
        return 0
    return (v_m1 - v_p1) / denom


def get_offset(ref, comp, search_radius=10):
    # Determine offset via cross-correlation
    rsh = ref.shape
    csh = comp.shape

    # No implementation if ref larger than comp
    assert rsh[0] <= csh[0] and rsh[1] <= csh[1]

    # No implementation for even shapes
    assert rsh[0] % 2 == 1 and rsh[1] % 2 == 1
    assert csh[0] % 2 == 1 and csh[1] % 2 == 1

    max_search_x = (csh[1] - rsh[1]) // 2
    max_search_y = (csh[0] - rsh[0]) // 2

    if search_radius > max_search_x:
        clip_x = search_radius - max_search_x
        ref = ref[:, clip_x:-clip_x]
    if search_radius > max_search_y:
        clip_y = search_radius - max_search_y
        ref = ref[clip_y:-clip_y, :]
    rsh = ref.shape

    corr = np.zeros((2 * search_radius + 1, 2 * search_radius + 1))
    for j, dy in enumerate(range(-search_radius, search_radius + 1)):
        for i, dx in enumerate(range(-search_radius, search_radius + 1)):
            ymin = (csh[0] - rsh[0]) // 2 + dy
            ymax = ymin + rsh[0]
            xmin = (csh[1] - rsh[1]) // 2 + dx
            xmax = xmin + rsh[1]
            crop = comp[ymin:ymax, xmin:xmax]

            corr[j, i] = np.sum(ref * crop) / np.sqrt(np.sum(ref**2) * np.sum(crop**2))

    best_ji = np.unravel_index(np.argmax(corr), corr.shape)
    best_offset = (best_ji[0] - search_radius, best_ji[1] - search_radius)

    if best_ji[0] == 0 or best_ji[0] == 2 * search_radius:
        return (np.nan, np.nan)
    if best_ji[1] == 0 or best_ji[1] == 2 * search_radius:
        return (np.nan, np.nan)

    dx_sub = get_parabola_vertex(
        corr[best_ji[0], best_ji[1] - 1],
        corr[best_ji[0], best_ji[1]],
        corr[best_ji[0], best_ji[1] + 1],
    )
    dy_sub = get_parabola_vertex(
        corr[best_ji[0] - 1, best_ji[1]],
        corr[best_ji[0], best_ji[1]],
        corr[best_ji[0] + 1, best_ji[1]],
    )

    best_offset = (best_offset[0] + dy_sub, best_offset[1] + dx_sub)

    return best_offset


def match_patches(img, ref, xs, ys, patch_size, search_radius):
    dx_out = []
    dy_out = []

    for x, y in zip(xs, ys):
        x = int(x)
        y = int(y)
        xmin = x - patch_size // 2
        xmax = x + patch_size // 2 + 1
        ymin = y - patch_size // 2
        ymax = y + patch_size // 2 + 1
        ref_crop = ref[ymin:ymax, xmin:xmax]
        if xmin < 0 or ymin < 0 or xmax > ref.shape[1] or ymax > ref.shape[1]:
            dy_out.append(np.nan)
            dx_out.append(np.nan)
            continue

        xmin_crop = xmin - search_radius
        xmax_crop = xmax + search_radius
        ymin_crop = ymin - search_radius
        ymax_crop = ymax + search_radius
        if (
            xmin_crop < 0
            or ymin_crop < 0
            or xmax_crop > img.shape[1]
            or ymax_crop > img.shape[0]
        ):
            dy_out.append(np.nan)
            dx_out.append(np.nan)
            continue
        img_crop = img[ymin_crop:ymax_crop, xmin_crop:xmax_crop]
        offset = get_offset(ref_crop, img_crop, search_radius)
        dy_out.append(offset[0])
        dx_out.append(offset[1])
    return dx_out, dy_out


def align_offsets(x, y, dx, dy):
    # Solve dx = dx_global - y dtheta_global
    #       dy = dy_global + x dtheta_global
    # for dx_global, dy_global, dtheta_global.
    design = np.zeros((2 * len(x), 3))
    design[: len(x), 0] = 1.0
    design[len(x) :, 1] = 1.0
    design[: len(x), 2] = -y
    design[len(x) :, 2] = x
    target = np.concatenate([dx, dy])
    w = np.isfinite(target)
    solution, *_ = np.linalg.lstsq(design[w], target[w])
    delta = design @ solution
    return dx - delta[: len(x)], dy - delta[len(x) :]


def stamp_sets_to_stamps(stamp_sets):
    all_stamps = []
    metadata = defaultdict(list)
    for stamp_set in stamp_sets:
        nstamp = 1 + len(stamp_set["tests"])
        all_stamps.append(stamp_set["ref"])
        all_stamps.extend(stamp_set["tests"])
        metadata["EXP_ID"].extend([stamp_set["ref_id"]] + stamp_set["test_ids"])
        metadata["DONUT_ID"].extend([stamp_set["donut_id"]] * nstamp)
        metadata["REF_ID"].extend([stamp_set["ref_id"]] * nstamp)

        x_ref_ccd_dvcs = stamp_set["x_ref_ccd_dvcs"].to_value(units.pix)
        y_ref_ccd_dvcs = stamp_set["y_ref_ccd_dvcs"].to_value(units.pix)
        x_ref_field_ccs = stamp_set["x_ref_field_ccs"].to_value(units.rad)
        y_ref_field_ccs = stamp_set["y_ref_field_ccs"].to_value(units.rad)
        x_ref_field_ocs = stamp_set["x_ref_field_ocs"].to_value(units.rad)
        y_ref_field_ocs = stamp_set["y_ref_field_ocs"].to_value(units.rad)

        metadata["X_REF"].extend([x_ref_ccd_dvcs] * nstamp)
        metadata["Y_REF"].extend([y_ref_ccd_dvcs] * nstamp)
        metadata["X_REF_CCS"].extend([float(x_ref_field_ccs)] * nstamp)
        metadata["Y_REF_CCS"].extend([float(y_ref_field_ccs)] * nstamp)
        metadata["X_REF_OCS"].extend([float(x_ref_field_ocs)] * nstamp)
        metadata["Y_REF_OCS"].extend([float(y_ref_field_ocs)] * nstamp)
        metadata["OFFSET_X"].extend(
            [0.0] + [off[1].to_value(units.pix) for off in stamp_set["offsets"]]
        )
        metadata["OFFSET_Y"].extend(
            [0.0] + [off[0].to_value(units.pix) for off in stamp_set["offsets"]]
        )
    out = Stamps(all_stamps)
    for key, val in metadata.items():
        out.metadata[key] = np.array(val)

    return out


def stamps_to_stamp_sets(stamps):
    if len(stamps) == 0:
        return []
    metadata = stamps.metadata
    donut_ids = np.array(metadata.getArray("DONUT_ID"))
    exp_ids = np.array(metadata.getArray("EXP_ID"))
    ref_ids = np.array(metadata.getArray("REF_ID"))

    stamp_sets = []
    for donut_id in np.unique(donut_ids):
        ref_donut_idx = np.where((donut_ids == donut_id) & (exp_ids == ref_ids))[0][0]
        ref = stamps[ref_donut_idx]
        test_idxs = np.where((donut_ids == donut_id) & (exp_ids != ref_ids))[0]
        tests = [stamps[i] for i in test_idxs]
        offset_x = [metadata.getArray("OFFSET_X")[i] for i in test_idxs]
        offset_y = [metadata.getArray("OFFSET_Y")[i] for i in test_idxs]
        offsets = list(zip(offset_y, offset_x))

        stamp_set = {}
        stamp_set["donut_id"] = donut_id
        stamp_set["ref"] = ref

        x_ref_ccd_dvcs = metadata.getArray("X_REF")[ref_donut_idx] * units.pix
        y_ref_ccd_dvcs = metadata.getArray("Y_REF")[ref_donut_idx] * units.pix
        x_ref_field_ccs = metadata.getArray("X_REF_CCS")[ref_donut_idx] * units.rad
        y_ref_field_ccs = metadata.getArray("Y_REF_CCS")[ref_donut_idx] * units.rad
        x_ref_field_ocs = metadata.getArray("X_REF_OCS")[ref_donut_idx] * units.rad
        y_ref_field_ocs = metadata.getArray("Y_REF_OCS")[ref_donut_idx] * units.rad

        stamp_set["x_ref_ccd_dvcs"] = x_ref_ccd_dvcs
        stamp_set["y_ref_ccd_dvcs"] = y_ref_ccd_dvcs
        stamp_set["x_ref_field_ccs"] = x_ref_field_ccs
        stamp_set["y_ref_field_ccs"] = y_ref_field_ccs
        stamp_set["x_ref_field_ocs"] = x_ref_field_ocs
        stamp_set["y_ref_field_ocs"] = y_ref_field_ocs
        stamp_set["tests"] = tests
        stamp_set["offsets"] = offsets
        stamp_set["ref_id"] = metadata.getArray("REF_ID")[ref_donut_idx]
        stamp_set["test_ids"] = [metadata.getArray("EXP_ID")[i] for i in test_idxs]

        stamp_sets.append(stamp_set)

    return stamp_sets


def fit_displacements(x, y, dx, dy):
    # Filter outliers in absolute displacement
    dr = np.hypot(dx, dy)
    quantiles = np.nanquantile(dr, [0.25, 0.5, 0.75])
    iqr = np.ptp(quantiles[[0, 2]])
    threshold = quantiles[1] + 3.0 * iqr

    flag_0 = dr < threshold

    # Fit Zks to inliers
    radius = np.hypot(x, y).max()
    zkBasis = zernikeBasis(28, x, y, R_outer=radius, R_inner=radius * 0.62)
    wgood = np.isfinite(dx) & np.isfinite(dy) & flag_0
    dx_coefs, *_ = np.linalg.lstsq(zkBasis.T[wgood], dx[wgood])
    dy_coefs, *_ = np.linalg.lstsq(zkBasis.T[wgood], dy[wgood])
    dx_fit = zkBasis.T @ dx_coefs
    dy_fit = zkBasis.T @ dy_coefs

    # Filter again on residuals to Zk fit
    ddx = dx - dx_fit
    ddy = dy - dy_fit
    xquant = np.nanquantile(ddx, [0.25, 0.5, 0.75])
    yquant = np.nanquantile(ddy, [0.25, 0.5, 0.75])
    xiqr = np.ptp(xquant[[0, 2]])
    yiqr = np.ptp(yquant[[0, 2]])
    xmed = xquant[1]
    ymed = yquant[1]
    xgood = (ddx > xmed - 3 * xiqr) & (ddx < xmed + 3 * xiqr)
    ygood = (ddy > ymed - 3 * yiqr) & (ddy < ymed + 3 * yiqr)
    good = np.isfinite(dx) & np.isfinite(dy) & xgood & ygood

    # Fit once more for the plot
    dx_coefs, *_ = np.linalg.lstsq(zkBasis.T[good], dx[good])
    dy_coefs, *_ = np.linalg.lstsq(zkBasis.T[good], dy[good])
    dx_fit = zkBasis.T @ dx_coefs
    dy_fit = zkBasis.T @ dy_coefs
    return dx_fit, dy_fit, good


class HartmannSensitivityAnalysisConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "detector")
):
    exposures = ct.Input(
        doc="Input exposure to make measurements on",
        dimensions=("exposure", "detector", "instrument"),
        storageClass="Exposure",
        name="post_isr_image",
        multiple=True,
        deferLoad=True,
    )
    user_groups = ct.Input(
        doc="Hartmann exposure groups",
        dimensions=("instrument",),
        storageClass="StructuredDataDict",
        name="hartmann_groups",
    )
    stamps = ct.Output(
        doc="Output Hartmann stamps",
        dimensions=("visit", "instrument", "detector"),
        storageClass="Stamps",
        name="hartmann_stamps",
        multiple=True,
    )
    detection_table = ct.Output(
        doc="Output Hartmann detection table",
        dimensions=("visit", "instrument", "detector"),
        storageClass="AstropyQTable",
        name="hartmann_detection_table",
        multiple=True,
    )
    analysis_table = ct.Output(
        doc="Output Hartmann sensitivity analysis table",
        dimensions=("visit", "instrument", "detector"),
        storageClass="AstropyQTable",
        name="hartmann_analysis_table",
        multiple=True,
    )
    unfiltered_plot = ct.Output(
        doc="Output Hartmann unfiltered plot",
        dimensions=("visit", "instrument", "detector"),
        storageClass="Plot",
        name="hartmann_unfiltered_plot",
        multiple=True,
    )
    filtered_plot = ct.Output(
        doc="Output Hartmann filtered plot",
        dimensions=("visit", "instrument", "detector"),
        storageClass="Plot",
        name="hartmann_filtered_plot",
        multiple=True,
    )
    residual_plot = ct.Output(
        doc="Output Hartmann residual plot",
        dimensions=("visit", "instrument", "detector"),
        storageClass="Plot",
        name="hartmann_residual_plot",
        multiple=True,
    )
    zernikes = ct.Output(
        doc="Output Hartmann Zernike coefficients",
        dimensions=("visit", "instrument", "detector"),
        storageClass="AstropyQTable",
        name="hartmann_zernikes",
        multiple=True,
    )
    exposure_table = ct.Output(
        doc="Output Hartmann exposure table",
        dimensions=("visit", "instrument", "detector"),
        storageClass="AstropyQTable",
        name="hartmann_exposure_table",
        multiple=True,
    )

    def __init__(self, *, config: Any | None = None) -> None:
        super().__init__(config=config)
        if config is not None:
            if not config.use_user_groups:
                del self.user_groups
                self.dimensions.add("group")


class HartmannSensitivityAnalysisConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=HartmannSensitivityAnalysisConnections,
):
    ref_index = pexConfig.Field[int](
        doc="Index of the reference exposure within the group",
        default=-1,
    )
    m2_dz = pexConfig.Field[float](
        doc="Defocus offset for M2 in mm",
        default=4.0,
    )
    cam_dz = pexConfig.Field[float](
        doc="Defocus offset for the camera in mm",
        default=4.0,
    )
    min_flux = pexConfig.Field[float](
        doc="Minimum flux for analysis",
        default=2e6,
    )
    max_inner_ratio = pexConfig.Field[float](
        doc="Maximum inner/total flux ratio for analysis",
        default=0.03,
    )
    max_outer_ratio = pexConfig.Field[float](
        doc="Maximum outer/total flux ratio for analysis",
        default=0.03,
    )
    do_plot = pexConfig.Field[bool](
        doc="Whether to make plots",
        default=True,
    )
    bin_size = pexConfig.Field[int](
        doc="Bin size for running initial detection",
        default=8,
    )
    rng_seed = pexConfig.Field[int](
        doc="Random number seed",
        default=57721,
    )
    n_pupil_radii = pexConfig.Field[int](
        doc="Number of pupil radii to probe",
        default=7,
    )
    zk_max = pexConfig.Field[int](
        doc="Maximum Zernike to fit",
        default=28,
    )
    max_donuts = pexConfig.Field[int](
        doc="Maximum number of donuts to analyze",
        default=3,
    )
    max_exp_plot = pexConfig.Field[int](
        doc="Maximum number of exposures to plot",
        default=4,
    )
    fea_dir = pexConfig.Field[str](
        doc="batoid_rubin fea_dir",
        default="/sdf/home/j/jmeyers3/.local/batoid_rubin_data",
    )
    bend_dir = pexConfig.Field[str](
        doc="batoid_rubin bend_dir",
        default="/sdf/home/j/jmeyers3/.local/batoid_rubin_data",
    )
    use_user_groups = pexConfig.Field[bool](
        doc="Whether to explicitly use user groups or to group by dimension",
        default=False,
    )
    isr = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="Instrument signature removal task",
    )
    subtractBackground = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )

    def setDefaults(self):
        self.isr.doAmpOffset = False
        self.isr.ampOffset.doApplyAmpOffset = False
        # Turn off slow steps in ISR but mask saturated pixels
        self.isr.doBrighterFatter = False
        self.isr.doSaturation = True
        self.isr.crosstalk.doQuadraticCrosstalkCorrection = False
        self.isr.qa.saveStats = False
        self.isr.doStandardStatistics = False
        self.isr.doInterpolate = False
        self.isr.doVariance = False
        self.isr.doDeferredCharge = False
        self.isr.doDefect = False
        self.isr.doApplyGains = True
        self.isr.doBias = False
        self.isr.doFlat = True
        self.isr.doDark = False
        self.isr.doLinearize = False
        self.isr.doSuspect = False
        self.isr.doSetBadRegions = False
        self.isr.doBootstrap = False
        self.isr.doCrosstalk = False
        self.isr.doITLEdgeBleedMask = False


class HartmannSensitivityAnalysis(
    pipeBase.PipelineTask,
):
    ConfigClass = HartmannSensitivityAnalysisConfig
    _DefaultName = "HartmannSensitivityAnalysis"

    def __init__(self, config, *, display=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.makeSubtask("isr")
        self.makeSubtask("subtractBackground")
        self._display = display

    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        # Group the input exposures
        if self.config.use_user_groups:
            exposure_handles = {v.dataId["exposure"]: v for v in inputRefs.exposures}
            groups = butlerQC.get(inputRefs.user_groups)
            exposure_groups = []
            for group in groups["groups"]:
                exposure_group = []
                for exp_id in group:
                    if exp_id in exposure_handles:
                        exposure_group.append(butlerQC.get(exposure_handles[exp_id]))
                    else:
                        self.log.warn(
                            "Exposure ID %s in group but not found in inputs", exp_id
                        )
                        break
                else:
                    exposure_groups.append(exposure_group)
        else:
            exposure_groups = [butlerQC.get(inputRefs.exposures)]

        # Run the analysis for each group
        # Write outputs using the visit ID of the reference exposure
        for igroup, exposure_handles in enumerate(exposure_groups):
            self.log.info(
                "Processing exposure group %d out of %d", igroup, len(exposure_groups)
            )
            exposures = [handle.get() for handle in exposure_handles]
            outputs = self.run(exposures)
            for key in outputRefs.keys():
                output = getattr(outputs, key)
                if output is None:
                    continue
                refs = getattr(outputRefs, key)
                for ref in refs:
                    if ref.dataId["visit"] == outputs.reference_exposure_id:
                        butlerQC.put(output, ref)
                        break

    def run(self, exposures, run_isr=False, **isr_kwargs):
        config = self.config

        self.log.info("Processing %d exposures in this group", len(exposures))
        reference_exposure, test_exposures = self.prepare_exposures(
            exposures, config.ref_index, run_isr=run_isr, **isr_kwargs
        )
        detections = self.detect(reference_exposure)

        if self._display is not None:
            self.update_display(reference_exposure, detections)

        stamp_sets, ref_telescope, test_telescopes, exposure_table = (
            self.get_initial_stamp_sets(detections, reference_exposure, test_exposures)
        )
        rtp = get_rtp(reference_exposure)
        detector = reference_exposure.getDetector()
        nrot = detector.getOrientation().getNQuarter()
        patch_table = self.match_all_patches(stamp_sets, ref_telescope, rtp, nrot)
        self.remove_net_shift_and_rotation(patch_table)
        self.fit_displacements(patch_table)
        self.predict_displacements(
            patch_table,
            detections,
            ref_telescope,
            test_telescopes,
            detector,
            rtp,
        )

        zernikes = self.fit_zernikes(patch_table, rtp, nrot)

        if self.config.do_plot:
            self.log.info("Making plots")
            initial_fig = self.plot_initial(stamp_sets, patch_table)
            filtered_fig = self.plot_filtered(stamp_sets, patch_table)
            resid_fig = self.plot_residual(stamp_sets, patch_table)
        else:
            initial_fig = None
            filtered_fig = None
            resid_fig = None

        return pipeBase.Struct(
            detection_table=detections,
            stamps=stamp_sets_to_stamps(stamp_sets),
            stamp_sets=stamp_sets,
            analysis_table=patch_table,
            unfiltered_plot=initial_fig,
            filtered_plot=filtered_fig,
            residual_plot=resid_fig,
            zernikes=zernikes,
            exposure_table=exposure_table,
            reference_exposure_id=reference_exposure.info.getVisitInfo().id,
        )

    def get_det_dz(self, exposure):
        detector = exposure.getDetector()
        orientation = detector.getOrientation()
        height = orientation.getHeight()
        height = -height  # because of DVCS to CCS
        return height

    def get_donut_radius(self, exposure):
        det_dz = self.get_det_dz(exposure)
        defocus = abs(self.config.m2_dz + self.config.cam_dz + det_dz)  # mm
        donut_diam = 85.0 * defocus  # Hardcoded to LSSTCam
        donut_radius = donut_diam / 2
        return donut_radius

    def prepare_exposures(self, exposures, ref_index, run_isr=False, **isr_kwargs):
        """Prepare exposures by running ISR (optional) and background subtraction.
        Sort exposures by visit ID and select reference exposure.
        """
        exposures.sort(key=lambda exp: exp.info.getVisitInfo().id)
        if run_isr:
            self.log.info("Running ISR on %d exposures", len(exposures))
            exposures = [self.isr.run(exp, **isr_kwargs).exposure for exp in exposures]
        self.log.info("Subtracting background")
        for exposure in exposures:
            self.subtractBackground.run(exposure=exposure)

        # Sort exposures and pick reference
        if ref_index < 0:  # Map -1 to N-1...
            ref_index = len(exposures) + ref_index
        reference_exposure = exposures[ref_index]
        ref_id = reference_exposure.info.getVisitInfo().id
        self.log.info("Using exposure %d as reference", ref_id)
        test_exposures = [exp for i, exp in enumerate(exposures) if i != ref_index]
        self.log.info("Using %d test exposures", len(test_exposures))
        for exp in test_exposures:
            test_id = exp.info.getVisitInfo().id
            self.log.info("  test exposure: %d", test_id)
        return reference_exposure, test_exposures

    def detect(self, exposure):
        """Detect donuts in the given exposure.
        Return a QTable
        """
        rtp = get_rtp(exposure)
        donut_radius = self.get_donut_radius(exposure)
        bin_size = self.config.bin_size
        binned_template_size = int(2 * donut_radius * 1.15 / bin_size)
        if binned_template_size % 2 == 0:
            binned_template_size += 1

        template = np.zeros((binned_template_size, binned_template_size), dtype=float)
        y, x = np.ogrid[
            -binned_template_size // 2 : binned_template_size // 2,
            -binned_template_size // 2 : binned_template_size // 2,
        ]
        r = np.hypot(x, y)
        binned_radius = donut_radius / bin_size
        template[r < binned_radius] = 1.0
        template[r < binned_radius * 0.62] = 0.0
        inner_hole = np.zeros_like(template)
        inner_hole[r < binned_radius * 0.55] = 1.0
        outer_annulus = np.zeros_like(template)
        outer_annulus[(r >= binned_radius * 1.05) & (r < binned_radius * 1.15)] = 1.0

        exp = exposure.clone()
        mi = exp.getMaskedImage()
        binned = afwMath.binImage(mi, bin_size)
        exp.setMaskedImage(binned)
        arr = exp.image.array
        mask = exp.mask.array

        # Histogram equalize since we care more about connected points above
        # threshold than actual flux values.
        cdf = np.nanquantile(arr, np.linspace(0, 1, 256))
        heq = np.digitize(arr, cdf)
        det = correlate(heq, template, mode="same")
        peaks = peak_local_max(
            det,
            min_distance=int(2 * 0.8 * binned_radius),
            exclude_border=int(binned_radius * 1.15),
        )

        table = QTable()
        table["donut_id"] = np.arange(len(peaks), dtype=np.int32)
        table["x_ref_ccd_dvcs"] = (peaks[:, 1] * bin_size).astype(np.int32) * units.pix
        table["y_ref_ccd_dvcs"] = (peaks[:, 0] * bin_size).astype(np.int32) * units.pix
        fluxes = []
        inner_fluxes = []
        outer_fluxes = []
        x_field_ccs_list = []
        y_field_ccs_list = []
        x_field_ocs_list = []
        y_field_ocs_list = []
        for peak in peaks:
            x_ccd_dvcs = peak[1] * bin_size
            y_ccd_dvcs = peak[0] * bin_size
            # Get OCS coordinates
            x_field_dvcs, y_field_dvcs = exposure.getDetector().transform(
                Point2D(x_ccd_dvcs, y_ccd_dvcs), PIXELS, FIELD_ANGLE
            )
            x_field_ccs, y_field_ccs = dvcs_to_ccs(x_field_dvcs, y_field_dvcs)
            x_field_ocs, y_field_ocs = ccs_to_ocs(x_field_ccs, y_field_ccs, rtp)
            x_field_ccs_list.append(x_field_ccs)
            y_field_ccs_list.append(y_field_ccs)
            x_field_ocs_list.append(x_field_ocs)
            y_field_ocs_list.append(y_field_ocs)

            xmin = peak[1] - binned_template_size // 2
            xmax = peak[1] + binned_template_size // 2 + 1
            ymin = peak[0] - binned_template_size // 2
            ymax = peak[0] + binned_template_size // 2 + 1
            stamp = arr[ymin:ymax, xmin:xmax]
            bad_mask_planes = ["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA"]
            bitmask = exp.mask.getPlaneBitMask(bad_mask_planes)
            msk = (mask[ymin:ymax, xmin:xmax] & bitmask) != 0

            use = xmin >= 0
            use = use and ymin >= 0
            use = use and xmax <= arr.shape[1]
            use = use and ymax <= arr.shape[0]
            use = use and np.sum(msk) == 0
            if not use:
                fluxes.append(np.nan)
                inner_fluxes.append(np.nan)
                outer_fluxes.append(np.nan)
                continue

            # Guess that a reasonable background estimate is the median of the
            # inner and outer pixels
            bkg = np.nanmedian(stamp * (np.maximum(inner_hole, outer_annulus)))
            stamp = stamp - bkg
            fluxes.append(np.nansum(stamp * template))
            inner_fluxes.append(np.nansum(stamp * inner_hole))
            outer_fluxes.append(np.nansum(stamp * outer_annulus))
        table["x_ref_field_ocs"] = (
            np.array(x_field_ocs_list, dtype=np.float32) * units.rad
        )
        table["y_ref_field_ocs"] = (
            np.array(y_field_ocs_list, dtype=np.float32) * units.rad
        )
        table["x_ref_field_ccs"] = (
            np.array(x_field_ccs_list, dtype=np.float32) * units.rad
        )
        table["y_ref_field_ccs"] = (
            np.array(y_field_ccs_list, dtype=np.float32) * units.rad
        )
        table["flux"] = np.array(fluxes, dtype=np.float32)
        table["inner_flux"] = np.array(inner_fluxes, dtype=np.float32)
        table["outer_flux"] = np.array(outer_fluxes, dtype=np.float32)
        table["inner_ratio"] = table["inner_flux"] / table["flux"]
        table["outer_ratio"] = table["outer_flux"] / table["flux"]
        table["use"] = table["flux"] > self.config.min_flux
        table["use"] &= table["inner_ratio"] < self.config.max_inner_ratio
        table["use"] &= table["outer_ratio"] < self.config.max_outer_ratio

        # Filter to max_donuts brightest
        subtable = table[table["use"]]
        subtable.sort(keys="flux", reverse=True)
        idxs = subtable["donut_id"][: self.config.max_donuts]
        table["use"][~np.isin(table["donut_id"], idxs)] = False

        self.log.info("Detected %d donuts", len(table))
        self.log.info("Selected %d donuts for analysis", np.sum(table["use"]))

        return table

    def get_initial_stamp_sets(self, detections, reference_exposure, test_exposures):
        if sum(detections["use"]) == 0:
            return [], None, [], None
        self.log.info("Connecting to EFD")
        efd_client = EFD()
        self.log.info("  Connected; Fetching states")
        ref_state = get_state(reference_exposure, efd_client)
        test_states = [get_state(exp, efd_client) for exp in test_exposures]
        self.log.info("  States fetched")
        exposure_table = QTable()
        # ID within the group.  -1 is the reference and 0..N-1 are the test cases.
        group_id = np.arange(-1, len(test_exposures), dtype=np.int32)
        exp_id = [reference_exposure.info.getVisitInfo().id]
        exp_id.extend([exp.info.getVisitInfo().id for exp in test_exposures])
        states = [ref_state]
        states.extend(test_states)
        exposure_table["group_id"] = group_id
        exposure_table["exp_id"] = exp_id
        exposure_table["state"] = states

        # Make optics
        band = reference_exposure.filter.bandLabel
        det_dz = self.get_det_dz(reference_exposure)
        ref_telescope = self.get_telescope(band, np.zeros_like(ref_state), det_dz)
        test_telescopes = [
            self.get_telescope(band, ts - ref_state, det_dz) for ts in test_states
        ]

        # Initial alignment of donuts
        stamp_sets = self.get_aligned_stamp_sets(
            reference_exposure,
            test_exposures,
            ref_telescope,
            test_telescopes,
            detections,
        )
        return stamp_sets, ref_telescope, test_telescopes, exposure_table

    def get_telescope(self, band, state, det_dz):
        # Handle sign flips and unit conversions
        state = np.array(state)
        state[[0, 1, 3, 5, 6, 8]] *= -1  # z and x are flipped
        state[30:] *= -1  # M2 modes are flipped
        state[[3, 4, 8, 9]] *= 3600  # deg to arcsec

        fiducial = (
            batoid.Optic.fromYaml(f"LSST_{band}.yaml")
            .withGloballyShiftedOptic("M2", [0, 0, self.config.m2_dz * 1e-3])
            .withGloballyShiftedOptic("LSSTCamera", [0, 0, self.config.cam_dz * 1e-3])
            .withGloballyShiftedOptic("Detector", [0, 0, det_dz * 1e-3])
        )
        builder = LSSTBuilder(
            fiducial,
            fea_dir=self.config.fea_dir,
            bend_dir=self.config.bend_dir,
        )
        builder = builder.with_aos_dof(state)
        return builder.build()

    def get_aligned_stamp_sets(
        self,
        reference_exposure,
        test_exposures,
        reference_telescope,
        test_telescopes,
        detections,
    ):
        self.log.info("Aligning detections")
        rtp = get_rtp(reference_exposure)
        donut_radius = self.get_donut_radius(reference_exposure)
        stamp_size = int(2 * donut_radius * 1.15)
        if stamp_size % 2 == 0:
            stamp_size += 1
        stamp_sets = []
        for detection in detections:
            if not detection["use"]:
                continue
            donut_id = detection["donut_id"]
            x_ref_ccd_dvcs = detection["x_ref_ccd_dvcs"].to_value(units.pix)
            y_ref_ccd_dvcs = detection["y_ref_ccd_dvcs"].to_value(units.pix)
            x_ref_field_ocs = detection["x_ref_field_ocs"].to_value(units.rad)
            y_ref_field_ocs = detection["y_ref_field_ocs"].to_value(units.rad)
            x_ref_field_ccs = detection["x_ref_field_ccs"].to_value(units.rad)
            y_ref_field_ccs = detection["y_ref_field_ccs"].to_value(units.rad)
            self.log.info(
                "  Aligning detection %d at (x,y)=(%d,%d)",
                donut_id,
                x_ref_ccd_dvcs,
                y_ref_ccd_dvcs,
            )

            cr = get_rays(
                reference_telescope, 0.0, 0.0, x_ref_field_ocs, y_ref_field_ocs
            )
            x_ref_predict, y_ref_predict = trace_ocs_to_ccd(
                cr.copy(),
                reference_telescope,
                reference_exposure.getDetector(),
                rtp,
            )
            x_ref_predict = x_ref_predict[0]
            y_ref_predict = y_ref_predict[0]

            # Extract stamp from reference
            xmin = x_ref_ccd_dvcs - stamp_size // 2
            xmax = x_ref_ccd_dvcs + stamp_size // 2 + 1
            ymin = y_ref_ccd_dvcs - stamp_size // 2
            ymax = y_ref_ccd_dvcs + stamp_size // 2 + 1
            box = Box2I(Point2I(xmin, ymin), Extent2I(xmax - xmin, ymax - ymin))
            try:
                ref_stamp = Stamp(reference_exposure.maskedImage[box])
            except LengthError:
                detection["use"] = False
                self.log.warn(
                    "  Could not extract reference stamp for detection %d; skipping",
                    donut_id,
                )
                continue
            ref_stamp_arr = ref_stamp.stamp_im.image.array

            offsets = []
            test_stamps = []
            for test_exposure, test_telescope in zip(test_exposures, test_telescopes):
                # Get expected position in test exposure
                x_test_predict, y_test_predict = trace_ocs_to_ccd(
                    cr.copy(),
                    test_telescope,
                    test_exposure.getDetector(),
                    rtp,
                )
                x_test_predict = x_test_predict[0]
                y_test_predict = y_test_predict[0]
                test_xmin = xmin + int(round(x_test_predict - x_ref_predict))
                test_xmax = xmax + int(round(x_test_predict - x_ref_predict))
                test_ymin = ymin + int(round(y_test_predict - y_ref_predict))
                test_ymax = ymax + int(round(y_test_predict - y_ref_predict))
                test_box = Box2I(
                    Point2I(test_xmin, test_ymin),
                    Extent2I(test_xmax - test_xmin, test_ymax - test_ymin),
                )
                if (
                    test_xmin < 0
                    or test_ymin < 0
                    or test_xmax > test_exposure.image.array.shape[1]
                    or test_ymax > test_exposure.image.array.shape[0]
                ):
                    detection["use"] = False
                    self.log.warn(
                        "  Aligned stamp for detection %d in exposure %d would be out of bounds; skipping",
                        donut_id,
                        test_exposure.info.getVisitInfo().id,
                    )
                    break
                test_stamp = Stamp(test_exposure.maskedImage[test_box])
                test_stamp_arr = test_stamp.stamp_im.image.array
                offset = get_offset(ref_stamp_arr, test_stamp_arr, search_radius=60)
                self.log.info(
                    "    Exposure %d: offset (y,x) = (%.2f, %.2f)",
                    test_exposure.info.getVisitInfo().id,
                    offset[0],
                    offset[1],
                )
                if not np.isfinite(offset).all():
                    detection["use"] = False
                    self.log.warn(
                        "  Could not align detection %d in exposure %d; skipping",
                        donut_id,
                        test_exposure.info.getVisitInfo().id,
                    )
                    break
                offsets.append(offset)
                test_box = Box2I(
                    Point2I(
                        test_xmin + int(round(offset[1])),
                        test_ymin + int(round(offset[0])),
                    ),
                    Extent2I(test_xmax - test_xmin, test_ymax - test_ymin),
                )
                try:
                    test_stamp = Stamp(test_exposure.maskedImage[test_box])
                except LengthError:
                    detection["use"] = False
                    self.log.warn(
                        "  Could not extract aligned stamp for detection %d in exposure %d; skipping",
                        donut_id,
                        test_exposure.info.getVisitInfo().id,
                    )
                    break
                test_stamps.append(test_stamp)
            if not detection["use"]:
                continue
            test_stamps = Stamps(test_stamps)
            stamp_sets.append(
                dict(
                    donut_id=donut_id,
                    ref=ref_stamp,
                    x_ref_ccd_dvcs=x_ref_ccd_dvcs * units.pix,
                    y_ref_ccd_dvcs=y_ref_ccd_dvcs * units.pix,
                    x_ref_field_ocs=x_ref_field_ocs * units.rad,
                    y_ref_field_ocs=y_ref_field_ocs * units.rad,
                    x_ref_field_ccs=x_ref_field_ccs * units.rad,
                    y_ref_field_ccs=y_ref_field_ccs * units.rad,
                    tests=test_stamps,
                    offsets=np.array(offsets) * units.pix,
                    ref_id=reference_exposure.info.getVisitInfo().id,
                    test_ids=[exp.info.getVisitInfo().id for exp in test_exposures],
                )
            )
        return stamp_sets

    def match_all_patches(self, stamp_sets, ref_telescope, rtp, nrot):
        patch_table = None
        for stamp_set in stamp_sets:
            self.log.info("Matching patches in donut %d", stamp_set["donut_id"])
            u, v, x, y = self.choose_pupil_locations(
                stamp_set, ref_telescope, rtp, nrad=self.config.n_pupil_radii, nrot=nrot
            )
            donut_patch_table = QTable()
            donut_patch_table["x"] = x
            donut_patch_table["y"] = y
            donut_patch_table["u"] = u
            donut_patch_table["v"] = v
            for iexp, test_stamp in enumerate(stamp_set["tests"]):
                self.log.info("  Matching in exposure %d", stamp_set["test_ids"][iexp])
                dx, dy = match_patches(
                    test_stamp.stamp_im.image.array,
                    stamp_set["ref"].stamp_im.image.array,
                    x.to_value(units.pix),
                    y.to_value(units.pix),
                    patch_size=30,
                    search_radius=10,
                )
                donut_patch_table[f"dx_{iexp}"] = dx * units.pix
                donut_patch_table[f"dy_{iexp}"] = dy * units.pix
            donut_patch_table["donut_id"] = stamp_set["donut_id"]
            if patch_table is None:
                patch_table = donut_patch_table
            else:
                patch_table = vstack([patch_table, donut_patch_table])
        return patch_table

    def choose_pupil_locations(self, stamp_set, telescope, rtp, nrad, nrot):
        dx, dy, model, zTA = fit_danish(
            telescope,
            x_ccs=stamp_set["x_ref_field_ccs"].to_value(units.rad),
            y_ccs=stamp_set["y_ref_field_ccs"].to_value(units.rad),
            stamp=stamp_set["ref"],
            nrot=nrot,
        )

        u, v = batoid.utils.hexapolar(
            outer=4.18,
            inner=4.18 * 0.612,
            nrad=nrad,
            naz=int(nrad * 6.28 / (1 - 0.612)),
        )
        x, y = pupil_to_pixel(u, v, dx, dy, zTA, rtp, nrot)

        return u * units.m, v * units.m, x * units.pix, y * units.pix

    def remove_net_shift_and_rotation(
        self,
        patch_table,
    ):
        if patch_table is None:
            return
        self.log.info("Removing net shifts and rotations")
        idxs = [int(col[3:]) for col in patch_table.colnames if col.startswith("dx_")]
        for idx in idxs:
            patch_table[f"dx_{idx}_aligned"] = np.nan * units.pix
            patch_table[f"dy_{idx}_aligned"] = np.nan * units.pix
        for donut_id in np.unique(patch_table["donut_id"]):
            wdonut = np.where(patch_table["donut_id"] == donut_id)[0]
            x = patch_table["x"][wdonut]
            y = patch_table["y"][wdonut]

            for idx in idxs:
                dx = patch_table[f"dx_{idx}"][wdonut]
                dy = patch_table[f"dy_{idx}"][wdonut]

                dx_aligned, dy_aligned = align_offsets(x, y, dx, dy)
                patch_table[f"dx_{idx}_aligned"][wdonut] = dx_aligned
                patch_table[f"dy_{idx}_aligned"][wdonut] = dy_aligned

    def predict_displacements(
        self,
        patch_table,
        detection_table,
        reference_telescope,
        test_telescopes,
        detector,
        rtp: Angle,
    ):
        if patch_table is None:
            return
        idxs = [
            int(col[3:4])
            for col in patch_table.colnames
            if col.startswith("dx_") and col.endswith("_aligned")
        ]
        for idx in idxs:
            patch_table[f"dx_{idx}_predict"] = np.nan * units.pix
            patch_table[f"dy_{idx}_predict"] = np.nan * units.pix

        for donut_id in np.unique(patch_table["donut_id"]):
            select = patch_table["donut_id"] == donut_id
            u = patch_table["u"][select].to_value(units.m)
            v = patch_table["v"][select].to_value(units.m)
            detection = detection_table[detection_table["donut_id"] == donut_id]
            x_field_ocs = detection["x_ref_field_ocs"][0].to_value(units.rad)
            y_field_ocs = detection["y_ref_field_ocs"][0].to_value(units.rad)
            rays = get_rays(
                reference_telescope,
                u,
                v,
                x_field_ocs,
                y_field_ocs,
            )
            x_ref, y_ref = trace_ocs_to_ccd(
                rays,
                reference_telescope,
                detector,
                rtp,
            )
            for idx, test_telescope in zip(idxs, test_telescopes):
                x_test, y_test = trace_ocs_to_ccd(
                    rays,
                    test_telescope,
                    detector,
                    rtp,
                )

                dx_test = x_test - x_ref
                dy_test = y_test - y_ref

                dx_test_align, dy_test_align = align_offsets(
                    x_ref, y_ref, dx_test, dy_test
                )

                patch_table[f"dx_{idx}_predict"][select] = dx_test_align * units.pix
                patch_table[f"dy_{idx}_predict"][select] = dy_test_align * units.pix

    def fit_zernikes(self, patch_table, rtp, nrot):
        if patch_table is None:
            return None
        donut_ids = np.unique(patch_table["donut_id"])
        ndonut = len(donut_ids)
        nexp = len(
            [
                col
                for col in patch_table.colnames
                if col.startswith("dx_") and col.endswith("_aligned")
            ]
        )
        zk_table = QTable()
        # Make columns
        zk_table["donut_id"] = donut_ids
        tmp = np.full(self.config.zk_max + 1, np.nan) * units.micron
        for iexp in range(nexp):
            zk_table[f"zk_{iexp}_ccs"] = [tmp] * ndonut
            zk_table[f"zk_{iexp}_ocs"] = [tmp] * ndonut
            zk_table[f"zk_{iexp}_ccs_predict"] = [tmp] * ndonut
            zk_table[f"zk_{iexp}_ocs_predict"] = [tmp] * ndonut

        for donut_id in donut_ids:
            wpatch = patch_table["donut_id"] == donut_id
            wzk = np.where(zk_table["donut_id"] == donut_id)[0][0]
            focal_length = 10.31 * units.m
            for iexp in range(nexp):
                use = patch_table[f"use_fit_{iexp}"][wpatch]
                dx = patch_table[f"dx_{iexp}_aligned"][wpatch][use]  # DVCS
                dy = patch_table[f"dy_{iexp}_aligned"][wpatch][use]
                dxp = patch_table[f"dx_{iexp}_predict"][wpatch][use]  # DVCS
                dyp = patch_table[f"dy_{iexp}_predict"][wpatch][use]
                u = patch_table["u"][wpatch][use]  # OCS
                v = patch_table["v"][wpatch][use]
                # Rotate DVCS -> CCS
                if nrot % 4 == 1:
                    dx, dy = -dy, dx
                    dxp
                elif nrot % 4 == 2:
                    dx, dy = -dx, -dy
                    dxp, dyp = -dxp, -dyp
                elif nrot % 4 == 3:
                    dx, dy = dy, -dx
                    dxp, dyp = dyp, -dxp
                dx, dy = dy, dx
                dxp, dyp = dyp, dxp
                # And CCS -> OCS
                dx, dy = ccs_to_ocs(dx, dy, rtp)
                dxp, dyp = ccs_to_ocs(dxp, dyp, rtp)
                A = np.hstack(
                    zernikeGradBases(
                        self.config.zk_max,
                        u.to_value(units.m),
                        v.to_value(units.m),
                        R_outer=4.18,
                        R_inner=4.18 * 0.612,  # meters
                    )
                ).T * focal_length.to_value(units.m)
                dx = dx * (10.0 * units.micron / units.pix)  # microns
                dy = dy * (10.0 * units.micron / units.pix)  # microns
                dxp = dxp * (10.0 * units.micron / units.pix)  # microns
                dyp = dyp * (10.0 * units.micron / units.pix)  # microns
                b = np.hstack([dx, dy])
                bp = np.hstack([dxp, dyp])
                zk_ocs, *_ = np.linalg.lstsq(A, b, rcond=None)
                rot = zernikeRotMatrix(self.config.zk_max, rtp.to_value(units.rad))
                zk_ccs = rot @ zk_ocs
                zk_ocs_predict, *_ = np.linalg.lstsq(A, bp, rcond=None)
                zk_ccs_predict = rot @ zk_ocs_predict

                zk_table[f"zk_{iexp}_ccs"][wzk] = zk_ccs
                zk_table[f"zk_{iexp}_ocs"][wzk] = zk_ocs
                zk_table[f"zk_{iexp}_ccs_predict"][wzk] = zk_ccs_predict
                zk_table[f"zk_{iexp}_ocs_predict"][wzk] = zk_ocs_predict

        return zk_table

    def update_display(
        self,
        exposure,
        donutCatalog,
    ):
        if self._display is None:
            raise RuntimeError("No display set")
        self._display.mtv(exposure)
        donut_radius = self.get_donut_radius(exposure)

        for idx, source in enumerate(donutCatalog):
            x, y = source["x_ref_ccd_dvcs"], source["y_ref_ccd_dvcs"]
            use = source["flux"] > self.config.min_flux
            use = use and source["inner_ratio"] < self.config.max_inner_ratio
            use = use and source["outer_ratio"] < self.config.max_outer_ratio
            color = "green" if use else "red"
            self._display.dot("o", x, y, size=donut_radius, ctype=color)
            self._display.dot(str(idx), x, y, ctype=color)

    def plot_initial(self, stamp_sets, patch_table):
        ndonut = len(stamp_sets)
        if ndonut == 0:
            nexp = 0
        else:
            nexp = len(stamp_sets[0]["tests"])

        figsize = (3 * self.config.max_exp_plot, 3 * self.config.max_donuts)
        fig = Figure(figsize=figsize, dpi=200)
        grispec_kw = dict(
            left=0.04, right=0.98, bottom=0.02, top=0.96, wspace=0.01, hspace=0.01
        )
        axs = fig.subplots(
            nrows=self.config.max_donuts,
            ncols=self.config.max_exp_plot,
            gridspec_kw=grispec_kw,
            squeeze=False,
        )
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for idonut in range(self.config.max_donuts):
            for iexp in range(self.config.max_exp_plot):
                ax = axs[idonut][iexp]
                if idonut >= ndonut or iexp >= nexp:
                    fig.delaxes(ax)
                    continue
                stamp_set = stamp_sets[idonut]
                if idonut == 0:
                    ax.set_title(stamp_set["test_ids"][iexp])
                if iexp == 0:
                    label = (
                        idonut,
                        (
                            int(stamp_set["x_ref_ccd_dvcs"].to_value(units.pix)),
                            int(stamp_set["y_ref_ccd_dvcs"].to_value(units.pix)),
                        ),
                    )
                    ax.set_ylabel(label)
                coords = patch_table[patch_table["donut_id"] == stamp_set["donut_id"]]
                x = np.array(coords["x"])
                y = np.array(coords["y"])
                dx = np.array(coords[f"dx_{iexp}_aligned"])
                dy = np.array(coords[f"dy_{iexp}_aligned"])

                ax = axs[idonut][iexp]
                vmax = np.quantile(stamp_set["ref"].stamp_im.image.array, 0.99)
                resid = (
                    stamp_set["ref"].stamp_im.image.array
                    - stamp_set["tests"][iexp].stamp_im.image.array
                )

                ax.imshow(
                    resid,
                    origin="lower",
                    cmap="bwr",
                    vmin=-vmax,
                    vmax=vmax,
                )
                Q = ax.quiver(
                    x,
                    y,
                    dx,
                    dy,
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                )
                ax.quiverkey(Q, 0.12, 0.88, 3, "3 pixels")
        return fig

    def fit_displacements(
        self,
        patch_table,
    ):
        if patch_table is None:
            return
        donut_ids = np.unique(patch_table["donut_id"])
        ndonut = len(donut_ids)
        nexp = len(
            [
                col
                for col in patch_table.colnames
                if col.startswith("dx_") and col.endswith("_aligned")
            ]
        )
        self.log.info(
            "Fitting displacements for %d donuts and %d exposures", ndonut, nexp
        )

        # Make room in table
        for iexp in range(nexp):
            patch_table[f"dx_{iexp}_fit"] = np.nan * units.pix
            patch_table[f"dy_{iexp}_fit"] = np.nan * units.pix
            patch_table[f"use_fit_{iexp}"] = False

        for donut_id in donut_ids:
            select = patch_table["donut_id"] == donut_id
            x = patch_table["x"][select].value
            y = patch_table["y"][select].value
            for iexp in range(nexp):
                dx = patch_table[f"dx_{iexp}_aligned"][select]
                dy = patch_table[f"dy_{iexp}_aligned"][select]
                dx_fit, dy_fit, use = fit_displacements(x, y, dx, dy)
                patch_table[f"dx_{iexp}_fit"][select] = dx_fit
                patch_table[f"dy_{iexp}_fit"][select] = dy_fit
                patch_table[f"use_fit_{iexp}"][select] = use

    def plot_filtered(self, stamp_sets, patch_table):
        ndonut = len(stamp_sets)
        if ndonut == 0:
            nexp = 0
        else:
            nexp = len(stamp_sets[0]["tests"])

        figsize = (3 * self.config.max_exp_plot, 3 * self.config.max_donuts)
        fig = Figure(figsize=figsize, dpi=200)
        grispec_kw = dict(
            left=0.04, right=0.98, bottom=0.02, top=0.96, wspace=0.01, hspace=0.01
        )
        axs = fig.subplots(
            nrows=self.config.max_donuts,
            ncols=self.config.max_exp_plot,
            gridspec_kw=grispec_kw,
            squeeze=False,
        )
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for idonut in range(self.config.max_donuts):
            for iexp in range(self.config.max_exp_plot):
                ax = axs[idonut][iexp]
                if idonut >= ndonut or iexp >= nexp:
                    fig.delaxes(ax)
                    continue
                stamp_set = stamp_sets[idonut]
                if idonut == 0:
                    ax.set_title(stamp_set["test_ids"][iexp])
                if iexp == 0:
                    label = (
                        idonut,
                        (
                            int(stamp_set["x_ref_ccd_dvcs"].to_value(units.pix)),
                            int(stamp_set["y_ref_ccd_dvcs"].to_value(units.pix)),
                        ),
                    )
                    ax.set_ylabel(label)
                coords = patch_table[patch_table["donut_id"] == stamp_set["donut_id"]]
                x = np.array(coords["x"])
                y = np.array(coords["y"])
                dx = np.array(coords[f"dx_{iexp}_aligned"])
                dy = np.array(coords[f"dy_{iexp}_aligned"])
                dx_predict = np.array(coords[f"dx_{iexp}_predict"])
                dy_predict = np.array(coords[f"dy_{iexp}_predict"])
                wgood = np.array(coords[f"use_fit_{iexp}"], dtype=bool)

                ax = axs[idonut][iexp]
                vmax = np.quantile(stamp_set["ref"].stamp_im.image.array, 0.99)
                resid = (
                    stamp_set["ref"].stamp_im.image.array
                    - stamp_set["tests"][iexp].stamp_im.image.array
                )

                ax.imshow(
                    resid,
                    origin="lower",
                    cmap="bwr",
                    vmin=-vmax,
                    vmax=vmax,
                )
                Q = ax.quiver(
                    x[wgood],
                    y[wgood],
                    dx[wgood],
                    dy[wgood],
                    color="k",
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                    width=0.002,
                )
                ax.quiver(
                    x[wgood],
                    y[wgood],
                    dx_predict[wgood],
                    dy_predict[wgood],
                    color="magenta",
                    width=0.002,
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                )
                ax.quiverkey(Q, 0.12, 0.88, 3, "3 pixels")
        return fig

    def plot_residual(self, stamp_sets, patch_table):
        ndonut = len(stamp_sets)
        if ndonut == 0:
            nexp = 0
        else:
            nexp = len(stamp_sets[0]["tests"])

        figsize = (3 * self.config.max_exp_plot, 3 * self.config.max_donuts)
        fig = Figure(figsize=figsize, dpi=200)
        grispec_kw = dict(
            left=0.04, right=0.98, bottom=0.02, top=0.96, wspace=0.01, hspace=0.01
        )
        axs = fig.subplots(
            nrows=self.config.max_donuts,
            ncols=self.config.max_exp_plot,
            gridspec_kw=grispec_kw,
            squeeze=False,
        )
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for idonut in range(self.config.max_donuts):
            for iexp in range(self.config.max_exp_plot):
                ax = axs[idonut][iexp]
                if idonut >= ndonut or iexp >= nexp:
                    fig.delaxes(ax)
                    continue
                stamp_set = stamp_sets[idonut]
                if idonut == 0:
                    ax.set_title(stamp_set["test_ids"][iexp])
                if iexp == 0:
                    label = (
                        idonut,
                        (
                            int(stamp_set["x_ref_ccd_dvcs"].to_value(units.pix)),
                            int(stamp_set["y_ref_ccd_dvcs"].to_value(units.pix)),
                        ),
                    )
                    ax.set_ylabel(label)
                coords = patch_table[patch_table["donut_id"] == stamp_set["donut_id"]]
                x = np.array(coords["x"])
                y = np.array(coords["y"])
                dx = np.array(coords[f"dx_{iexp}_aligned"])
                dy = np.array(coords[f"dy_{iexp}_aligned"])
                dx_predict = np.array(coords[f"dx_{iexp}_predict"])
                dy_predict = np.array(coords[f"dy_{iexp}_predict"])
                dx_resid = dx - dx_predict
                dy_resid = dy - dy_predict
                wgood = np.array(coords[f"use_fit_{iexp}"], dtype=bool)

                ax = axs[idonut][iexp]
                vmax = np.quantile(stamp_set["ref"].stamp_im.image.array, 0.99)
                resid = (
                    stamp_set["ref"].stamp_im.image.array
                    - stamp_set["tests"][iexp].stamp_im.image.array
                )

                ax.imshow(
                    resid,
                    origin="lower",
                    cmap="bwr",
                    vmin=-vmax,
                    vmax=vmax,
                )
                Q = ax.quiver(
                    x[wgood],
                    y[wgood],
                    dx_resid[wgood],
                    dy_resid[wgood],
                    color="k",
                    scale_units="xy",
                    angles="xy",
                    scale=0.1,
                    pivot="middle",
                    width=0.002,
                )
                ax.quiverkey(Q, 0.12, 0.88, 3, "3 pixels")
        return fig
