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

import lsst.pipe.base as pipeBase
from lsst.daf.butler import DataCoordinate

from .aggregate_donut_tables_cwfs import (
    AggregateDonutTablesCwfsTask,
    AggregateDonutTablesCwfsTaskConnections,
)
from .utilities import intra_focal_ids

__all__ = [
    "AggregateDonutTablesCwfsFamTaskConnections",
    "AggregateDonutTablesCwfsFamTaskConfig",
    "AggregateDonutTablesCwfsFamTask",
]


class AggregateDonutTablesCwfsFamTaskConnections(
    AggregateDonutTablesCwfsTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    def adjust_all_quanta(self, adjuster: pipeBase.QuantaAdjuster) -> None:
        """Move each intra-focal detector's donutTables ref into the
        following visit's quantum, then drop the emptied quantum.
        Classification is snapshotted up front since `get_inputs`
        reflects live state and would otherwise re-flag an
        extra-focal visit as intra-focal once it receives moved refs.

        Notes
        -----
        This task relies on the convention that the intra-focal and
        extra-focal exposures are consecutive visits (the extra-focal
        visit is ``visit + 1`` of the intra-focal visit). A visit is
        treated as intra-focal when it contains donutTables from the
        intra-focal detectors, and those refs are moved into the paired
        extra-focal (``visit + 1``) quantum.

        Because of this, the data query must select the intra-focal
        detectors from the intra-focal visit and the extra-focal detectors
        from the extra-focal visit, e.g.::

            -d "instrument='LSSTCam' and (
                    (visit.id=<intra_visit>
                     and detector.id in (192,196,200,204))
                 or (visit.id=<extra_visit>
                     and detector.id in (191,195,199,203)))"

        If instead both visits are selected on their own (without
        restricting the detectors), the extra-focal visit will also
        contain intra-focal donutTables. It is then classified as
        intra-focal and resolves to a paired visit (``visit + 1``) that
        does not exist in the quantum graph, and a `RuntimeError` will be
        raised. Restrict each visit to the correct set of detectors as
        shown above to avoid this.
        """
        to_do = set(adjuster.iter_data_ids())
        intra_refs_by_data_id = {
            data_id: intra_refs
            for data_id in to_do
            if (
                intra_refs := [
                    ref
                    for ref in adjuster.get_inputs(data_id)["donutTables"]
                    if ref.dataId["detector"] in intra_focal_ids
                ]
            )
        }

        for data_id, intra_refs in intra_refs_by_data_id.items():
            # The way RA runs the custom QG builder will ensure the
            # extra focal quantum has the intra focal input.
            extra_focal_data_id = DataCoordinate.standardize(data_id, visit=int(data_id["visit"]) + 1)

            if extra_focal_data_id not in to_do:
                raise RuntimeError(
                    f"Could not find the extra-focal visit {extra_focal_data_id} paired with "
                    f"intra-focal visit {data_id}. Restrict each visit to the correct detectors "
                    "(intra-focal visit to the intra-focal detectors, extra-focal visit to the "
                    "extra-focal detectors); see the docstring for an example."
                )

            for ref in intra_refs:
                adjuster.add_input(extra_focal_data_id, "donutTables", ref)

            adjuster.remove_quantum(data_id)


class AggregateDonutTablesCwfsFamTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=AggregateDonutTablesCwfsFamTaskConnections,  # type: ignore
):
    pass


class AggregateDonutTablesCwfsFamTask(AggregateDonutTablesCwfsTask):
    ConfigClass = AggregateDonutTablesCwfsFamTaskConfig  # type: ignore[assignment]
    _DefaultName = "AggregateDonutTablesCwfsFam"
