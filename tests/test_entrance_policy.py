import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.entrance_decision import decide
from benchmark.entrances import measure
from benchmark.oblique_recess import rectify_wall


class EntrancePolicyTests(unittest.TestCase):
    def test_review_rim_never_changes_incumbent_or_asserts_depth(self):
        base = [(np.array([0, 0]), np.array([2, 3]))]
        extra = [(np.array([4, 0]), np.array([6, 3]))]
        wall = dict(rect=True, grid=[7, 3], calibration={}, H=np.eye(3))
        with patch('benchmark.entrances.propose', return_value=base), \
             patch('benchmark.entrances.agreed_extras', return_value=extra), \
             patch('benchmark.entrances.measure_boxes', return_value=[dict(depth=2)]) as measured:
            result = measure(wall, 'dark', review_extras=True)
        measured.assert_called_once_with(wall, base, base)
        original, additional = result['candidates']
        self.assertEqual(original, dict(depth=2))
        self.assertEqual(additional['status'], 'review_only')
        self.assertIsNone(additional['depth'])

    def test_no_pavement_abstains_before_proposing_openings(self):
        wall = rectify_wall(Image.new('RGB', (256, 160), 'gray'))
        result = measure(wall, 'dark', review_extras=True)
        self.assertEqual(result['reason'], 'no stable pavement calibration')
        self.assertEqual(result['candidates'], [])

    def test_promotion_requires_gain_without_more_errors_on_either_layout(self):
        candidate = dict(selected='new', incumbent='old', promotion_rule='test')
        def report(exact, false=0):
            return dict(selected='new', metrics={
                'old': dict(real=4, exact=2, wrong_depth=0, false_recess=0),
                'new': dict(real=4, exact=exact, wrong_depth=0, false_recess=false)})
        for validation, confirmation, expected in [
            (report(2), report(2), 'old'), (report(2), report(3), 'new'),
            (report(3), report(2, 1), 'old'), (report(3), report(1), 'old')
        ]:
            self.assertEqual(decide(candidate, validation, confirmation)['selected'], expected)
