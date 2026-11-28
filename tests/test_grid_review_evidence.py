import unittest

from benchmark.grid_review_evidence import grid_preference


class GridReviewEvidenceTests(unittest.TestCase):
    def test_visibly_warped_grid_is_resolved_without_a_vote(self):
        corner = dict(covered=85.4, cells=3.17, phase=.26, faces=49)
        earlier = dict(covered=97.2, cells=1.01, phase=.05, faces=49)
        result = grid_preference(corner, earlier)
        self.assertEqual(result["preferred"], "B")
        self.assertEqual(result["evidence_for_b"], ["coverage", "corner alignment", "block scale"])

    def test_coverage_and_alignment_can_disagree(self):
        corner = dict(covered=99.5, cells=1.03, phase=.07, faces=70)
        earlier = dict(covered=88.5, cells=1.03, phase=.02, faces=70)
        result = grid_preference(corner, earlier)
        self.assertIsNone(result["preferred"])
        self.assertEqual(result["evidence_for_a"], ["coverage"])
        self.assertEqual(result["evidence_for_b"], ["corner alignment"])

    def test_small_differences_do_not_pick_a_winner(self):
        corner = dict(covered=99.9, cells=.99, phase=.04, faces=80)
        earlier = dict(covered=99.9, cells=1, phase=.01, faces=80)
        self.assertIsNone(grid_preference(corner, earlier)["preferred"])

    def test_missing_reference_faces_abstain(self):
        self.assertIsNone(grid_preference({"faces": 2}, {"faces": 2})["preferred"])

    def test_degenerate_grid_abstains(self):
        broken = dict(covered=80, cells=0, phase=.15, faces=20)
        valid = dict(covered=95, cells=1, phase=.03, faces=20)
        self.assertIsNone(grid_preference(broken, valid)["preferred"])
