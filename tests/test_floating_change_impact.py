import unittest
from benchmark.floating_change_impact import plan


class FloatingChangeImpact(unittest.TestCase):
    def test_viewer_change_reuses_scoring(self):
        impact = plan(["benchmark/floating_review_app.py"])
        self.assertEqual(list(impact["refresh"]), ["site"])

    def test_grid_change_requires_downstream_checks(self):
        impact = plan(["benchmark/native_surface_counter.py", "engine/new_dependency.py"])
        self.assertIn("assembly", impact["refresh"])
        self.assertIn("perfect", impact["refresh"])
        self.assertNotIn("fields", impact["refresh"])
        self.assertEqual(impact["needsClassification"], ["engine/new_dependency.py"])

    def test_perfect_presentation_change_is_conservative(self):
        self.assertNotIn("score", plan(["benchmark/floating_perfect_report.py"])["refresh"])
