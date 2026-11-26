"""Camera-space surface lifting checked against projected, known-size boxes."""
import unittest

import numpy as np

from engine.surface_model import assemble, lift


class SurfaceModelTests(unittest.TestCase):
    def setUp(self):
        self.size = (960, 640)
        self.focal = 505.
        self.center = np.array(self.size) / 2
        self.K = np.array([[self.focal, 0, self.center[0]], [0, self.focal, self.center[1]], [0, 0, 1.]])
        yaw, pitch = np.radians([31., -15.])
        ct, st = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        self.rotation = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array(
            [[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        self.front = self.in_camera([(0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)])
        self.side = self.in_camera([(3, 0, 0), (3, 0, 2), (3, 2, 2), (3, 2, 0)])

    def in_camera(self, points):
        return np.asarray(points) @ self.rotation.T + [.4, -.6, 12.]

    def image(self, xyz):
        return xyz[:, :2] * self.focal / xyz[:, 2, None] + self.center

    def model(self, faces, counts, diagnostic=True):
        planes = [dict(id=label, quad=self.image(xyz).tolist()) for label, xyz in faces]
        pitches = [dict(counts=n) for n in counts]
        return assemble(planes, pitches, self.size, self.K if diagnostic else None)

    def test_lift_recovers_depth_normal_and_lengths_with_known_intrinsics(self):
        points, normal, distance = lift(self.image(self.side), self.K, dict(counts=[2, 2]))
        np.testing.assert_allclose(points*distance, self.side, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(normal), 1., atol=1e-12)
        np.testing.assert_allclose(normal @ np.diff(self.side, axis=0).T, 0., atol=1e-12)
        self.assertLess(normal @ self.side.mean(0), 0.)
        np.testing.assert_allclose(lift(self.image(self.side), self.K, dict(counts=None))[0], points)
        self.assertIsNone(lift(self.image(self.side), self.K, dict(counts=[2, 5]))[2])

    def test_shared_edge_transfers_block_scale_to_all_four_return_corners(self):
        result = self.model([('front', self.front), ('return', self.side)], [[3, 2], None])
        self.assertEqual(result['lens_source'], 'recorded-camera-diagnostic')
        a, b = result['planes']
        self.assertEqual((a['units'], a['scale_source']), ('blocks', 'local-pitch'))
        self.assertEqual((b['units'], b['scale_source']), ('blocks', 'shared-edge'))
        self.assertEqual(a['component'], b['component'])
        [edge] = result['shared_edges']
        np.testing.assert_allclose(np.array(a['corners_camera'])[edge['corners'][0]],
                                   np.array(b['corners_camera'])[edge['corners'][1]], atol=1e-9)
        np.testing.assert_allclose(a['corners_camera'], self.front, atol=1e-9)
        np.testing.assert_allclose(b['corners_camera'], self.side, atol=1e-9)

    def test_relative_component_reconstructs_box_up_to_one_common_scale(self):
        result = self.model([('front', self.front), ('return', self.side)], [None, None], diagnostic=False)
        self.assertEqual(result['lens_source'], 'orthogonal-outline-vanishing-points')
        np.testing.assert_allclose(result['intrinsics'], self.K, atol=1e-8)
        self.assertEqual(len(result['shared_edges']), 1)
        for p, xyz in zip(result['planes'], (self.front, self.side)):
            self.assertEqual(p['units'], 'relative')
            np.testing.assert_allclose(np.array(p['corners_camera'])*self.front[:, 2].mean(), xyz, atol=1e-8)

    def test_coplanar_seam_and_inconsistent_scales_keep_components_separate(self):
        flat = self.in_camera([(3, 0, 0), (6, 0, 0), (6, 2, 0), (3, 2, 0)])
        result = self.model([('left', self.front), ('right', flat)], [[3, 2], None])
        self.assertEqual(result['shared_edges'], [])
        self.assertEqual([p['units'] for p in result['planes']], ['blocks', 'relative'])
        self.assertNotEqual(*[p['component'] for p in result['planes']])
        conflict = self.model([('front', self.front), ('return', self.side)], [[3, 2], [4, 4]])
        self.assertEqual(conflict['shared_edges'], [])
        self.assertNotEqual(*[p['component'] for p in conflict['planes']])

    def test_parallel_recess_back_depth_requires_its_own_scale(self):
        back = self.in_camera([(0, 0, 2), (3, 0, 2), (3, 2, 2), (0, 2, 2)])
        faces = [('wall', self.front), ('back', back)]
        unsupported = self.model(faces, [[3, 2], None])
        self.assertEqual(unsupported['shared_edges'], [])
        self.assertEqual(unsupported['planes'][1]['units'], 'relative')
        anchored = self.model(faces, [[3, 2], [3, 2]])
        coords = np.array([p['corners_camera'] for p in anchored['planes']])
        expected = np.broadcast_to(2*self.rotation[:, 2], (4, 3))
        np.testing.assert_allclose(coords[1] - coords[0], expected, atol=1e-9)
