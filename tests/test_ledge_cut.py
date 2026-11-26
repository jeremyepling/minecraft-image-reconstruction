import unittest

import numpy as np

from benchmark.ledge_cut_test import corner_evidence


class LedgeJunctionTest(unittest.TestCase):
    def test_source_pixel_junctions_in_both_axes_and_directions(self):
        # Adjacent fourth-pixel sample centers fall at y=38 and 42; their boundary is y=40.
        samples = np.array([[7, x, 9, x, 10, x, 12, x] for x in range(5, 17)])
        main = [[22, 40], [66, 40]]
        cases = {
            'L': ([main, [[66, 40], [66, 65]]], True),
            'T': ([main, [[66, 15], [66, 65]]], True),
            'X': ([main, [[66, 40], [92, 40]], [[66, 15], [66, 65]]], False),
            'straight only': ([main], False),
        }
        for name, (segments, expected) in cases.items():
            for transpose in (False, True):
                for reverse in (False, True):
                    with self.subTest(name=name, transpose=transpose, reverse=reverse):
                        pts = samples.reshape(-1, 4, 2)
                        lines = np.asarray(segments)
                        if transpose:
                            pts = pts[:, :, ::-1]
                            lines = lines[:, :, ::-1]
                        if reverse:
                            pts = pts[:, ::-1]
                            lines = lines[:, ::-1]
                        support, _ = corner_evidence(pts.reshape(-1, 8), lines)
                        self.assertEqual(support is not None, expected)
