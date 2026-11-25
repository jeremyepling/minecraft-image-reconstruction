import unittest

import numpy as np

from benchmark.ratio_metrics import score
from calibration.world import GROUND
from engine.geometry import project


def wall(oblique=True, gap=False):
    w, h, f = 800, 600, 640
    origin = np.array([0., GROUND + 1, 0.])
    eye = origin + ([12, 8, 16] if oblique else [4, 2, 16])
    forward = origin + [4, 2, 0] - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    native = np.eye(4)
    native[:, 3] = 0
    native[0, :3], native[1, :3], native[3, :3] = 2*f/w*right, 2*f/h*up, forward
    pose = dict(view_rotation_projection=native.T.ravel().tolist(), image_size=[w, h], eye_world=eye.tolist())
    linear = np.stack([right*f + w/2*forward, -up*f + h/2*forward, forward])
    P = np.column_stack((linear, linear @ (origin - eye)))
    prediction = dict(P=P.tolist(), axes_homogeneous=P[:, :3].T.tolist(), image_size=[w, h],
                      scale_status="supported", planes=[dict(fixed_axis=2, axes=[0, 1], offset=0,
                                                             span_blocks=[8, 4])])
    faces = []
    for x in range(8):
        for y in range(4):
            if gap and x == 4:
                continue
            corners = np.array([[x, y, 0], [x+1, y, 0], [x+1, y+1, 0], [x, y+1, 0]])
            faces.append(dict(normal=[0, 0, 1], world_corners=(corners + origin).astype(int).tolist(),
                              image_corners=project(P, corners).tolist(), corner_visible=[True]*4, visible_fraction=1.))
    return pose, prediction, faces


class RatioScoringTest(unittest.TestCase):
    def test_perspective_correction_uses_only_pixels_and_vanishing_points(self):
        _, prediction, faces = wall()
        prediction.update(P=None, planes=[], scale_status="uncertain")
        ratios = score(prediction, {}, faces)["endpoint_diagnostic"]
        self.assertEqual(ratios["measured"], 4)
        self.assertLess(ratios["mean_abs_error_pct"], 1.e-7)
        for case in ratios["cases"]:
            self.assertAlmostEqual(case["predicted_ratio"], 2.)
        naive = [np.linalg.norm(np.subtract(c["width_end_px"], c["corner_px"])) /
                 np.linalg.norm(np.subtract(c["height_end_px"], c["corner_px"])) for c in ratios["cases"]]
        self.assertGreater(max(abs(np.asarray(naive) - 2.)), .2)

    def test_ratio_survives_unit_and_homogeneous_rescaling(self):
        pose, prediction, faces = wall()
        prediction["axes_homogeneous"] = (np.array(prediction["axes_homogeneous"]) *
                                           np.array([-.1, 7.7, -300.])[:, None]).tolist()
        prediction["planes"][0]["span_blocks"] = [24, 12]
        prediction["scale_status"] = "uncertain"
        rows = score(prediction, pose, faces)
        self.assertLess(rows["endpoint_diagnostic"]["mean_abs_error_pct"], 1.e-7)
        automatic = rows["automatic_charts"]
        self.assertEqual(automatic["measured"], 1)
        self.assertEqual(automatic["mean_abs_error_pct"], 0.)
        self.assertIsNone(automatic["cases"][0]["plane_error_blocks"])

    def test_front_parallel_ratio_and_no_measurements_cross_open_gap(self):
        _, prediction, faces = wall(oblique=False, gap=True)
        prediction.update(P=None, planes=[])
        rows = score(prediction, {}, faces)["endpoint_diagnostic"]
        self.assertEqual(rows["measured"], 8)
        self.assertLess(rows["mean_abs_error_pct"], 1.e-7)
        for case in rows["cases"]:
            x1, x2 = sorted((case["corner_xyz"][0], case["width_end_xyz"][0]))
            self.assertFalse(x1 <= 4 and x2 >= 5)

    def test_no_axes_no_metric_coverage_and_no_guessed_counts(self):
        _, prediction, faces = wall()
        prediction.update(P=None, planes=[], axes_homogeneous=[], scale_status="uncertain")
        rows = score(prediction, {}, faces)
        self.assertEqual(rows["endpoint_diagnostic"]["abstained"], 4)
        self.assertEqual(rows["automatic_charts"]["abstained"], 1)
        for group in rows.values():
            self.assertIsNone(group["mean_abs_error_pct"])
            self.assertTrue(all(case["predicted_ratio"] is None for case in group["cases"]))
