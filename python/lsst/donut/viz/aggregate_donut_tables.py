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
from astropy.table import vstack

import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, Camera
from lsst.afw.image import VisitInfo
from lsst.geom import Point2D, radians
from lsst.pipe.base import connectionTypes as ct
from lsst.ts.wep.utils import convertDictToVisitInfo
from lsst.utils.timer import timeMethod

__all__ = [
    "AggregateDonutTablesTaskConnections",
    "AggregateDonutTablesTaskConfig",
    "AggregateDonutTablesTask",
]


# Note: cannot make visit a dimension because we have not yet paired visits.
class AggregateDonutTablesTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    donutTablesIntra = ct.Input(
        doc="Donut tables intra",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutTableIntra",
        multiple=True,
    )
    donutTablesExtra = ct.Input(
        doc="Donut tables extra",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutTableExtra",
        multiple=True,
    )
    qualityTables = ct.Input(
        doc="Donut quality tables",
        dimensions=("visit", "detector", "instrument"),
        storageClass="AstropyQTable",
        name="donutQualityTable",
        multiple=True,
        deferGraphConstraint=True,
    )
    camera = ct.PrerequisiteInput(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    aggregateDonutTable = ct.Output(
        doc="Visit-level table of donuts and Zernikes",
        dimensions=("visit", "instrument"),
        storageClass="AstropyQTable",
        name="aggregateDonutTable",
    )


class AggregateDonutTablesTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesTaskConnections,  # type: ignore
):
    pass


class AggregateDonutTablesTask(pipeBase.PipelineTask):
    ConfigClass = AggregateDonutTablesTaskConfig
    _DefaultName = "AggregateDonutTables"

    @timeMethod
    def runQuantum(
        self,
        butlerQC: pipeBase.QuantumContext,
        inputRefs: pipeBase.InputQuantizedConnection,
        outputRefs: pipeBase.OutputQuantizedConnection,
    ) -> None:
        camera = butlerQC.get(inputRefs.camera)

        # Make dictionaries to match visits and detectors
        donutTablesIntra = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.donutTablesIntra}
        donutTablesExtra = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.donutTablesExtra}
        qualityTables = {ref.dataId["detector"]: butlerQC.get(ref) for ref in inputRefs.qualityTables}

        donutTableIntra = next(iter(donutTablesIntra.values()))
        donutTableExtra = next(iter(donutTablesExtra.values()))
        extraVisitInfo = convertDictToVisitInfo(donutTableExtra.meta["visit_info"])
        intraVisitInfo = convertDictToVisitInfo(donutTableIntra.meta["visit_info"])

        result = self.run(
            camera, intraVisitInfo, extraVisitInfo, donutTablesIntra, donutTablesExtra, qualityTables
        )
        butlerQC.put(result.out, outputRefs.aggregateDonutTable)

    @timeMethod
    def run(
        self,
        camera: Camera,
        intraVisitInfo: VisitInfo,
        extraVisitInfo: VisitInfo,
        donutTablesIntra: dict,
        donutTablesExtra: dict,
        qualityTables: dict,
    ) -> pipeBase.Struct:
        """Aggregate donut tables for a set of visits.

        Parameters
        ----------
        camera : lsst.afw.cameraGeom.Camera
            The camera object.
        intraVisitInfo : VisitInfo
            Visit info object for intra-focal exposure.
        extraVisitInfo : VisitInfo
            Visit info object for extra-focal exposure.
        donutTablesIntra : dict
            Dictionary of intra-focal donut tables keyed by detector.
        donutTablesExtra : dict
            Dictionary of extra-focal donut tables keyed by detector.
        qualityTables : dict
            Dictionary of quality tables keyed by detector.

        Returns
        -------
        struct
            Struct of aggregated donut tables, keyed on extra-focal visit.
        """
        # Find common detectors between donut and quality tables
        # DonutQualityTables only saved under extra-focal ids
        detectors = donutTablesExtra.keys() & donutTablesIntra.keys() & qualityTables.keys()

        # Raise error if there's no matches
        if len(detectors) == 0:
            raise RuntimeError("No detector matches found between the donut and quality tables")

        tables = []

        # Iterate over the common (visit, detector) pairs
        for detector in detectors:
            # Get pixels -> field angle transform for this detector
            det = camera[detector]
            tform = det.getTransform(PIXELS, FIELD_ANGLE)

            # Load the donut catalog table, and the donut quality table
            intraDonutTable = donutTablesIntra[detector]
            extraDonutTable = donutTablesExtra[detector]
            qualityTable = qualityTables[detector]

            # Get rows of quality table for this exposure
            intraQualityTable = qualityTable[qualityTable["DEFOCAL_TYPE"] == "intra"]
            extraQualityTable = qualityTable[qualityTable["DEFOCAL_TYPE"] == "extra"]

            if (len(extraQualityTable) == 0) or (len(intraQualityTable) == 0):
                continue

            for donutTable, qualityTable in zip(
                [intraDonutTable, extraDonutTable],
                [intraQualityTable, extraQualityTable],
            ):
                # Select donuts used in Zernike estimation
                table = donutTable[qualityTable["FINAL_SELECT"]]

                # Add focusZ to donut table
                table["focusZ"] = table.meta["visit_info"]["focus_z"]

                # Add SN from quality table to the donut table
                table["snr"] = qualityTable["SN"][qualityTable["FINAL_SELECT"]]

                # Add field angle in CCS to the table
                pts = tform.applyForward(
                    [Point2D(x, y) for x, y in zip(table["centroid_x"], table["centroid_y"])]
                )
                table["thx_CCS"] = [pt.y for pt in pts]  # Transpose from DVCS to CCS
                table["thy_CCS"] = [pt.x for pt in pts]
                table["detector"] = det.getName()

                tables.append(table)

        # Don't attempt to stack metadata
        for table in tables:
            table.meta = {}

        out = vstack(tables)

        # Add metadata for extra and intra focal exposures
        # TODO: Swap parallactic angle for pseudo parallactic angle.
        #       See SMTN-019 for details.
        out.meta["extra"] = {
            "visit": extraVisitInfo.id,
            "focusZ": extraVisitInfo.focusZ,
            "parallacticAngle": extraVisitInfo.boresightParAngle.asRadians(),
            "rotAngle": extraVisitInfo.boresightRotAngle.asRadians(),
            "rotTelPos": extraVisitInfo.boresightParAngle.asRadians()
            - extraVisitInfo.boresightRotAngle.asRadians()
            - np.pi / 2,
            "ra": extraVisitInfo.boresightRaDec.getRa().asRadians(),
            "dec": extraVisitInfo.boresightRaDec.getDec().asRadians(),
            "az": extraVisitInfo.boresightAzAlt.getLongitude().asRadians(),
            "alt": extraVisitInfo.boresightAzAlt.getLatitude().asRadians(),
            "mjd": extraVisitInfo.date.toAstropy().mjd,
        }
        out.meta["intra"] = {
            "visit": intraVisitInfo.id,
            "focusZ": intraVisitInfo.focusZ,
            "parallacticAngle": intraVisitInfo.boresightParAngle.asRadians(),
            "rotAngle": intraVisitInfo.boresightRotAngle.asRadians(),
            "rotTelPos": intraVisitInfo.boresightParAngle.asRadians()
            - intraVisitInfo.boresightRotAngle.asRadians()
            - np.pi / 2,
            "ra": intraVisitInfo.boresightRaDec.getRa().asRadians(),
            "dec": intraVisitInfo.boresightRaDec.getDec().asRadians(),
            "az": intraVisitInfo.boresightAzAlt.getLongitude().asRadians(),
            "alt": intraVisitInfo.boresightAzAlt.getLatitude().asRadians(),
            "mjd": intraVisitInfo.date.toAstropy().mjd,
        }

        # Carefully average angles in meta
        out.meta["average"] = {}
        for k in (
            "parallacticAngle",
            "rotAngle",
            "rotTelPos",
            "ra",
            "dec",
            "az",
            "alt",
        ):
            a1 = out.meta["extra"][k] * radians
            a2 = out.meta["intra"][k] * radians
            a2 = a2.wrapNear(a1)
            out.meta["average"][k] = ((a1 + a2) / 2).wrapCtr().asRadians()

        # Easier to average the MJDs
        out.meta["average"]["mjd"] = 0.5 * (out.meta["extra"]["mjd"] + out.meta["intra"]["mjd"])

        # Calculate coordinates in different reference frames
        q = out.meta["average"]["parallacticAngle"]
        rtp = out.meta["average"]["rotTelPos"]
        out["thx_OCS"] = np.cos(rtp) * out["thx_CCS"] - np.sin(rtp) * out["thy_CCS"]
        out["thy_OCS"] = np.sin(rtp) * out["thx_CCS"] + np.cos(rtp) * out["thy_CCS"]
        out["th_N"] = np.cos(q) * out["thx_CCS"] - np.sin(q) * out["thy_CCS"]
        out["th_W"] = np.sin(q) * out["thx_CCS"] + np.cos(q) * out["thy_CCS"]

        return pipeBase.Struct(out=out)
