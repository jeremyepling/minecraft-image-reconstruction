import unittest

import numpy as np

from benchmark.depth_probe_score import occlusion_pairs
from benchmark.normal_round import normal_diagnostic, relation_score, support_gap
from benchmark.surface_metrics import DOWNSAMPLE, score_planes
from benchmark.surface_topology import opening_coverage, opening_mask
from engine.surface_order import order


SIZE = (96, 64)


def vertical_panels(depths=(5, 9)):
    panels = []
    for i, (name, z) in enumerate(zip(("left", "right"), depths)):
        x = 40 * i + 8
        quad = [[x, 8], [x + 40, 8], [x + 40, 56], [x, 56]]
        corners = [[(u - 48) * z / 64, (v - 32) * z / 64, z] for u, v in quad]
        panels.append(dict(id=name, quad=quad, corners_camera=corners, units="blocks", component=i))
    return panels


def support_runs(mask):
    runs = []
    for y, row in enumerate(mask):
        ends = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)))
        runs.extend([y, int(x0), int(x1)] for x0, x1 in ends.reshape(-1, 2))
    return {"size": list(mask.shape[::-1]), "runs": runs}


class NormalRoundMetricsTest(unittest.TestCase):
    def test_geometry_orders_equal_scales_in_both_directions_and_abstains_on_ties(self):
        for depths, expected in (((5, 9), ("left", "right")), ((9, 5), ("right", "left"))):
            with self.subTest(depths=depths):
                panels = vertical_panels(depths)
                for planes in (panels, panels[::-1]):
                    relation, = order(planes, SIZE)
                    self.assertEqual((relation["near"], relation["far"]), expected)
                    self.assertGreaterEqual(relation["samples"], 4)
                    self.assertEqual(relation["agreement"], 1.)
        self.assertEqual(order(vertical_panels((7, 7)), SIZE), [])

    def test_separate_relative_scales_abstain_but_connected_relative_planes_order(self):
        panels = vertical_panels()
        panels[1]["units"] = "relative"
        self.assertEqual(order(panels, SIZE), [])
        panels[0]["units"] = "relative"
        self.assertEqual(order(panels, SIZE), [])
        panels[1]["component"] = panels[0]["component"]
        relation, = order(panels, SIZE)
        self.assertEqual((relation["near"], relation["far"]), ("left", "right"))

    def test_tilted_plane_is_nearer_at_the_shared_silhouette_than_its_center_indicates(self):
        left, right = vertical_panels((7, 6.4))
        # Project a parallelogram with constant world height, receding toward its left edge.
        left["quad"] = [[8, 20], [48, 10.4], [48, 53.6], [8, 44]]
        left["corners_camera"] = [[(u - 48) * z / 64, (v - 32) * z / 64, z]
                                  for (u, v), z in zip(left["quad"], (9, 5, 5, 9))]
        relation, = order([left, right], SIZE)
        self.assertEqual((relation["near"], relation["far"]), ("left", "right"))

    def test_inverse_depth_samples_both_scanline_axes_without_scale(self):
        panels = [{"id": p["id"], "quad": p["quad"]} for p in vertical_panels()]
        field = np.zeros(SIZE[::-1], np.float32)
        field[:, :48] = 2
        self.assertEqual(order(panels, SIZE), [])
        for size, depth, parts in ((SIZE, field, panels),
                                  (SIZE[::-1], field.T.copy(), [
                                      dict(id=p["id"], quad=[[y, x] for x, y in p["quad"]]) for p in panels])):
            for sign, expected in ((1, ("left", "right")), (-1, ("right", "left"))):
                rel, = order(parts, size, sign * depth)
                self.assertEqual((rel["near"], rel["far"]), expected)
            self.assertEqual(order(parts, size, np.zeros_like(depth)), [])

    def test_truth_depth_discontinuities_and_classification_with_unmatched_proposal(self):
        ids = np.empty((12, 36), int)
        depths = (4., 8., 5.)
        field = np.empty_like(ids, dtype=np.float32)
        for i, z in enumerate(depths):
            ids[:, 12*i:12*(i+1)] = i + 1
            field[:, 12*i:12*(i+1)] = 1/z
        t = dict(id_map=ids, charts={i+1: np.diag([1., 1., z]) for i, z in enumerate(depths)})
        reference = occlusion_pairs(t, field)
        self.assertEqual((reference["eligible_pairs"], reference["correct_order"]), (2, 2))
        self.assertEqual(occlusion_pairs(t, -field)["correct_order"], 0)
        parts = [{"id": s} for s in ("a", "b", "c", "unmatched")]
        matches = [dict(proposal_index=i, patch_id=i+1) for i in range(3)]
        relations = [dict(near=a, far=b) for a, b in (
            ("a", "b"), ("b", "c"), ("a", "c"), ("unmatched", "a"))]
        result = relation_score(relations, parts, matches, reference)
        self.assertEqual(tuple(result[k] for k in ("eligible", "proposed", "correct", "reversed", "other")),
                         (2, 4, 1, 1, 2))
        empty = relation_score([], parts, matches, reference)
        self.assertEqual((empty["eligible"], empty["proposed"], empty["correct"]), (2, 0, 0))
        restricted = relation_score(relations, parts, matches, reference, eligible_patches={1, 2})
        self.assertEqual(tuple(restricted[k] for k in ("eligible", "correct", "reversed", "other")),
                         (1, 1, 0, 3))

    def test_opposite_truth_orders_on_separate_segments_never_double_count_one_proposal(self):
        # The scorer pools observations by patch ID; a global proposed order cannot get two verdicts.
        truth_pairs = dict(pairs=[dict(near_patch=1, far_patch=2), dict(near_patch=2, far_patch=1)])
        result = relation_score([dict(near="left", far="right")], vertical_panels(),
                                [dict(proposal_index=0, patch_id=1), dict(proposal_index=1, patch_id=2)],
                                truth_pairs)
        self.assertEqual(tuple(result[k] for k in ("eligible", "proposed", "correct", "reversed", "other")),
                         (0, 1, 0, 0, 1))

    def test_true_open_arch_with_same_contour_but_different_observed_support(self):
        ids = np.zeros((8, 12), int)
        ids[1:7, 1:11] = 1
        ids[2:7, 2:10] = 0
        gap = opening_mask(ids)
        self.assertEqual(int(gap.sum()), 40)
        size = (ids.shape[1] * DOWNSAMPLE, ids.shape[0] * DOWNSAMPLE)
        outline = dict(quad=[[4, 4], [44, 4], [44, 28], [4, 28]])
        mask = np.repeat(np.repeat(ids > 0, 2, axis=0), 2, axis=1)
        arch = dict(**outline, support=support_runs(mask))
        rectangular = opening_coverage(gap, [outline])
        self.assertEqual(support_gap(gap, [outline], rectangular, size)["filled_pixels"], 40)
        self.assertEqual(support_gap(gap, [arch], rectangular, size)["filled_pixels"], 0)
        intrusion = dict(quad=[[16, 12], [24, 12], [24, 20], [16, 20]])
        mixed = [arch, intrusion, intrusion]
        self.assertEqual(support_gap(gap, mixed, opening_coverage(gap, mixed), size)["filled_pixels"], 4)
        t = dict(id_map=ids, image_size=size, charts={1: np.diag([1., 1., 5.])}, patches={
            1: dict(eligible=True, support_px=int(np.count_nonzero(ids)), normal_camera=[0, 0, -1])})
        self.assertEqual(score_planes(t, [arch])["matched"], 0)
        result = score_planes(t, [arch], supported_pixels=True)
        self.assertEqual((result["matched"], result["spill_pixels"], result["matches"][0]["iou"]), (1, 0, 1.))

    def test_normal_accuracy_is_signed_and_counts_soffits_separately(self):
        ids = np.ones((4, 8), int)
        ids[:, 4:] = 2
        t = dict(id_map=ids, patches={
            1: dict(normal=[0, 0, 1], normal_camera=[0, 0, -1]),
            2: dict(normal=[0, -1, 0], normal_camera=[0, 1, 0])})
        normal = np.empty((8, 16, 3), np.float32)
        normal[:, :8] = [0, 0, -1]
        normal[:, 8:] = [0, 1, 0]
        good = normal_diagnostic(t, normal, [32, 16])
        self.assertEqual((good["samples"], good["within_15"], good["soffit_within_15"]), (32, 32, 16))
        normal[:, 8:] *= -1
        flipped = normal_diagnostic(t, normal, [32, 16])
        self.assertEqual((flipped["within_15"], flipped["soffit_samples"], flipped["soffit_within_15"]),
                         (16, 16, 0))
