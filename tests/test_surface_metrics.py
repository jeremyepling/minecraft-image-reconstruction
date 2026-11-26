import unittest

import numpy as np

from benchmark.camera_labels import face_labels
from benchmark.surface_metrics import _one_to_one, score_pitch, score_planes, truth
from calibration.world import GROUND
from engine.geometry import project, transform


def wall_scene(extra=()):
    y0 = GROUND+1
    m = np.diag([1., 1., 1., 0.])
    m[3, 2] = -1
    pose = dict(id="synthetic", image_size=[256,192], eye_world=[2.,y0+1.5,8.],
                view_rotation_projection=m.T.ravel().tolist())
    blocks = {(x, y0+y, 0): dict(block_id="stone", state="stone") for x in range(4) for y in range(3)}
    blocks.update({p: dict(block_id="tuff", state="tuff") for p in extra})
    faces = face_labels(pose, blocks, {p: "wall" for p in blocks})
    return truth(pose, faces)


class SurfaceMetricsTest(unittest.TestCase):
    def test_one_to_one_coverage_and_spill(self):
        t = wall_scene()
        pid = t["rectangles"][0]["patch_id"]
        q = project(t["charts"][pid], [[0,GROUND+4],[4,GROUND+4],[4,GROUND+1],[0,GROUND+1]]).tolist()
        match = score_planes(t, [{"quad":q}])
        self.assertEqual((match["eligible"],match["matched"]), (1,1))
        self.assertEqual(match["matches"][0]["normal_camera"],[0,0,-1])
        self.assertAlmostEqual(match["matches"][0]["depth_at_center"],7.)
        self.assertEqual(match["matches"][0]["iou"],1.)
        self.assertEqual(match["spill_pixels"],0)
        extra = dict(quad=[[4,4],[28,4],[28,20],[4,20]])
        spilt = score_planes(t, [{"quad":q}, {"quad":q}, extra])
        self.assertEqual(spilt["matched"],1)
        self.assertAlmostEqual(spilt["precision"],1/3)
        self.assertEqual(spilt["recall"],1)
        self.assertEqual(spilt["spill_pixels"],24)
        self.assertEqual(spilt["proposed_pixels"],match["proposed_pixels"]+24)
        self.assertEqual(_one_to_one([(.5,0,1),(.5,0,2),(.5,1,1)]), [(0,2),(1,1)])

    def test_occluding_face_keeps_nearest_id_and_blocks_true_rectangle(self):
        t = wall_scene(extra=[(1,GROUND+2,2)])
        front = next(p for p in t["patches"].values() if p["axis"] == 2 and p["offset"] == 3)
        back = next(p for p in t["patches"].values() if p["axis"] == 2 and p["offset"] == 1)
        x, y = (project(t["charts"][front["id"]], [[1.5,GROUND+2.5]])[0]/4).astype(int)
        self.assertEqual(t["id_map"][y,x],front["id"])
        self.assertFalse(front["eligible"])
        self.assertTrue(back["eligible"])
        self.assertEqual(back["rectangle"]["block_bounds"],[2,GROUND+1,4,GROUND+4])

    def test_disconnected_coplanar_cells_are_distinct(self):
        t = wall_scene(extra=[(7,GROUND+1,0)])
        south = [p for p in t["patches"].values() if p["normal"] == [0,0,1]]
        self.assertEqual(sorted(p["faces"] for p in south),[1,12])
        self.assertEqual(sum(p["eligible"] for p in south),1)

    def test_pitch_uses_subblock_extent_and_observed_joint_phase(self):
        t = wall_scene()
        chart = t["rectangles"][0]
        pid, q = chart["patch_id"], chart["quad"]
        counts = np.array(chart["counts"])
        np.testing.assert_allclose(counts,[3.57,2.58])
        raw = project(np.linalg.inv(transform(q)), t["vertices"][pid])
        phase = (raw[0]*counts) % 1.
        exact = score_pitch(t, pid, q, dict(counts=counts,grid=dict(phase_blocks=phase)))
        self.assertAlmostEqual(exact["mean_count_error"],0.)
        self.assertTrue(exact["half_block_both"])
        self.assertEqual(exact["truth_intersections"],6)
        self.assertEqual((exact["grid_precision"],exact["grid_recall"],exact["grid_f1"]), (1.,1.,1.))
        shifted = score_pitch(t, pid, q, dict(counts=counts,grid=dict(phase_blocks=phase+[.5,0.])))
        self.assertTrue(shifted["half_block_both"])
        self.assertEqual((shifted["grid_precision"],shifted["grid_recall"]), (0.,0.))
        missing = score_pitch(t, pid, q, {"counts":None,"grid":None})
        self.assertFalse(missing["half_block_both"])
        self.assertEqual(missing["grid_f1"],0.)
        self.assertIsNone(missing["mean_count_error"])

    def test_measured_proposal_span_follows_its_own_edge_order(self):
        t = wall_scene()
        c = t["rectangles"][0]
        q = np.roll(c["quad"], 1, axis=0)
        ans = score_pitch(t, c["patch_id"], q, dict(counts=[2.58,3.57],grid=None))
        np.testing.assert_allclose(ans["expected_counts"],[2.58,3.57])
        self.assertAlmostEqual(ans["mean_count_error"],0.)
