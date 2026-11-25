import unittest

import numpy as np

from benchmark.camera_metrics import score, truth_camera
from benchmark.camera_round import FROZEN, read
from calibration.world import GROUND


class CameraScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = next(f for f in read(FROZEN / 'index.json')['frames'] if f['split'] == 'dev')
        cls.pose = read(FROZEN / frame['pose'])
        cls.faces = read(FROZEN / frame['labels'])['faces']

    def camera(self, shift=0.):
        # A different horizontal orientation and distant arbitrary integer origin.
        transform = np.array([[0, 0, -1, 26 + shift], [0, 1, 0, GROUND + 1],
                              [1, 0, 0, -32], [0, 0, 0, 1]])
        P = truth_camera(self.pose) @ transform
        return dict(P=P.tolist(), axes_homogeneous=P[:, :3].T.tolist(), planes=[])

    def test_perfect_camera_and_visible_spans_ignore_integer_gauge(self):
        scores = score(self.camera(), self.pose, self.faces)
        self.assertGreater(scores['vertices'], 100)
        self.assertGreater(scores['spans'], 20)
        self.assertEqual(scores['vertices'], scores['within_2px'])
        self.assertLess(scores['mean_px'], 1e-6)
        self.assertLess(scores['span_mae'], 1e-6)

    def test_registration_cannot_hide_fractional_block_shift(self):
        scores = score(self.camera(.49), self.pose, self.faces)
        self.assertLess(scores['hit_rate'], .5)
        self.assertGreater(scores['mean_px'], 4.)

    def test_abstention_preserves_denominators(self):
        scores = score(dict(P=None, planes=[]), self.pose, self.faces)
        self.assertEqual(scores['within_2px'], 0)
        self.assertGreater(scores['vertices'], 100)
        self.assertGreater(scores['spans'], 20)
        self.assertIsNone(scores['span_mae'])
