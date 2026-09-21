# This file is part of donut_viz.
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

import numpy as np
from astropy import units as u
from astropy.table import vstack

import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera
from lsst.geom import Point2D
from lsst.ts.wep.utils import convertDictToVisitInfo
from lsst.utils.timer import timeMethod

from .aggregate_donut_tables_cwfs import (
    AggregateDonutTablesCwfsTask,
    AggregateDonutTablesCwfsTaskConnections,
)

__all__ = [
    "AggregateDonutTablesUnpairedTaskConfig",
    "AggregateDonutTablesUnpairedTask",
]


class AggregateDonutTablesUnpairedTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesCwfsTaskConnections,  # type: ignore
):
    pass


class AggregateDonutTablesUnpairedTask(AggregateDonutTablesCwfsTask):
    ConfigClass = AggregateDonutTablesUnpairedTaskConfig  # type: ignore[assignment]
    _DefaultName = "AggregateDonutTablesUnpaired"

    @timeMethod
    def run(
        self,
        camera: Camera,
        donutTables: dict,
        qualityTables: dict,
    ) -> pipeBase.Struct:
        """Aggregate donut tables for a set of visits.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.Camera
            The camera object.
        donutTables : dict
            Dictionary of donut tables keyed by detector.
        qualityTables : dict
            Dictionary of quality tables keyed by detector.

        Returns
        -------
        struct
            Struct of aggregated donut tables, keyed on extra-focal visit.
        """
        tables = []
        extraDetectorIds = [191, 195, 199, 203]
        intraDetectorIds = [192, 196, 200, 204]

        for detector in donutTables.keys():
            if detector not in qualityTables.keys():
                continue

            det = camera[detector]

            # Load the donut catalog table, and the donut quality table
            donutTable = donutTables[detector]
            qualityTable = qualityTables[detector]

            if len(qualityTable) == 0:
                continue

            table = donutTable[qualityTable["FINAL_SELECT"]]

            # Add focusZ to donut table
            offset = (
                1.5 if det.getId() in extraDetectorIds else -1.5 if det.getId() in intraDetectorIds else 0
            )
            table["focusZ"] = table.meta["visit_info"]["focus_z"] + offset * u.mm

            # Add SN from quality table to the donut table
            table["snr"] = qualityTable["SN"][qualityTable["FINAL_SELECT"]]

            # Get pixels -> field angle transform for this detector
            tform = det.getTransform(PIXELS, FIELD_ANGLE)

            # Add field angle in CCS to the table
            pts = tform.applyForward(
                [Point2D(x, y) for x, y in zip(table["centroid_x"], table["centroid_y"])]
            )
            table["thx_CCS"] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
            table["thy_CCS"] = [pt.x for pt in pts]
            table["detector"] = det.getName()

            tables.append(table)

        # Grab visitInfo. The last one will do since all should be the same.
        visitInfo = convertDictToVisitInfo(table.meta["visit_info"])

        # Don't attempt to stack metadata
        for table in tables:
            table.meta = {}

        out = vstack(tables)

        # Add metadata for extra and intra focal exposures
        # TODO: Swap parallactic angle for pseudo parallactic angle.
        #       See SMTN-019 for details.
        out.meta["visitInfo"] = {
            "visit": visitInfo.id,
            "focusZ": visitInfo.focusZ,
            "parallacticAngle": visitInfo.boresightParAngle.asRadians(),
            "rotAngle": visitInfo.boresightRotAngle.asRadians(),
            "rotTelPos": visitInfo.boresightParAngle.asRadians()
            - visitInfo.boresightRotAngle.asRadians()
            - np.pi / 2,
            "ra": visitInfo.boresightRaDec.getRa().asRadians(),
            "dec": visitInfo.boresightRaDec.getDec().asRadians(),
            "az": visitInfo.boresightAzAlt.getLongitude().asRadians(),
            "alt": visitInfo.boresightAzAlt.getLatitude().asRadians(),
            "mjd": visitInfo.date.toAstropy().mjd,
        }

        # Calculate coordinates in different reference frames
        q = out.meta["visitInfo"]["parallacticAngle"]
        rtp = out.meta["visitInfo"]["rotTelPos"]
        out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
        out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
        out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
        out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

        return pipeBase.Struct(aggregateDonutTable=out)
