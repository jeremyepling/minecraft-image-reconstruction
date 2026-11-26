import unittest
from unittest.mock import patch

from benchmark.known_spacing_atlas import predict


class KnownSpacingAtlasTest(unittest.TestCase):
    @patch('benchmark.known_spacing_atlas.template_block_seam.predict')
    def test_requires_two_positive_measured_spacings(self, register):
        for bad in (None, 0, -1, float('nan'), float('inf')):
            for counts in ([bad, 2.4], [1.8, bad]):
                baseline = dict(counts=counts, phase=[.4, None])
                self.assertEqual(predict({}, baseline), baseline)
        register.assert_not_called()

    @patch('benchmark.known_spacing_atlas.template_block_seam.predict')
    def test_preserves_counts_known_offset_and_inputs(self, register):
        register.return_value = dict(counts=[1.5, 2.39], phase=[.25, .75])
        baseline = dict(counts=[1.6, 2.4], phase=[.23, None])
        self.assertEqual(predict({}, baseline), dict(counts=[1.6, 2.4], phase=[.23, .75]))
        self.assertEqual(baseline, dict(counts=[1.6, 2.4], phase=[.23, None]))

    @patch('benchmark.known_spacing_atlas.template_block_seam.predict')
    def test_already_located_or_missing_crop_needs_no_registration(self, register):
        baseline = dict(counts=[4.2, 2.7], phase=[.3, .8])
        self.assertEqual(predict({}, baseline), baseline)
        baseline['phase'][0] = None
        self.assertEqual(predict(None, baseline), baseline)
        register.assert_not_called()
