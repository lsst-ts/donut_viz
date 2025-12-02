from lsst.donut.viz.utilities import (
    add_coordinate_roses,
    get_day_obs_seq_num_from_visitid,
    get_instrument_channel_name,
)
from lsst.utils.tests import TestCase


class TestDonutVizUtilities(TestCase):
    def testGetInstrumentChannelName(self) -> None:
        self.assertTrue(get_instrument_channel_name("LSSTCam"), "lsstcam_aos")
        self.assertTrue(get_instrument_channel_name("LSSTCamSim"), "lsstcam_sim_aos")
        self.assertTrue(get_instrument_channel_name("LSSTComCam"), "comcam_aos")
        self.assertTrue(get_instrument_channel_name("LSSTComCamSim"), "comcam_sim_aos")
        with self.assertRaises(ValueError) as context:
            get_instrument_channel_name("LSSTCAM")
        expected_msg = "Unknown instrument LSSTCAM"
        self.assertEqual(str(context.exception), expected_msg)

    def testGetDayObsSeqNumFromVisitId(self) -> None:
        day_obs, seq_num = get_day_obs_seq_num_from_visitid(4021123106001)
        self.assertEqual(day_obs, 20211231)
        self.assertEqual(seq_num, 6001)

    def testAddCoordinateRoses(self) -> None:
        with self.assertRaises(ValueError):
            add_coordinate_roses(None, None, None, [1, 2])
        with self.assertRaises(ValueError) as context:
            add_coordinate_roses(None, None, None, [(1, 2)])
        expected_msg = "If p0 is not None, it must be a pair of (x, y) coordinates"
        self.assertEqual(str(context.exception), expected_msg)
