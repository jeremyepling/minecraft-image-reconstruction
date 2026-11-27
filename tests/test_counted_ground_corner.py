import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import counted_ground_corner as corner
from engine.plane_consensus import row_runs
from engine.surface_model import intrinsic
from engine.visible_grid import _blocks
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE, rasterize


class CountedGroundCornerTests(unittest.TestCase):
    def setUp(self):
        self.size = (640, 480)
        self.image = Image.new('RGB', self.size)
        self.eye = np.array([-9., 7., -12.])
        forward = np.array([1., 2., 1.]) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(right, forward), forward])
        self.projection = intrinsic(self.size, 660) @ np.column_stack((rotation, -rotation @ self.eye))
        self.masks = []
        self.parts = []
        self.charts = []
        for axis in (0, 2):
            quad = np.zeros((4, 3))
            quad[:, 1] = [5, 5, 0, 0]
            quad[:, 2-axis] = [0, 4, 4, 0]
            uvw = quad @ self.projection[:, :3].T + self.projection[:, 3]
            quad = uvw[:, :2] / uvw[:, 2:3]
            if quad[0, 0] > quad[1, 0]:
                quad = quad[[1, 0, 3, 2]]
            shape = (self.size[1]//SAMPLE_STRIDE, self.size[0]//SAMPLE_STRIDE)
            mask = np.zeros(shape, bool)
            uvw[:, :2] /= SAMPLE_STRIDE
            box, pixels, *_ = rasterize(uvw, shape[1], shape[0])
            mask[box] = pixels
            normal = np.zeros(3)
            normal[axis] = -1
            key = str(axis)
            self.parts.append(dict(id=key, normal_camera=rotation @ normal))
            self.charts.append(dict(id=key, pitch=dict(counts=[4., 5.]), quad=quad.tolist()))
            self.masks.append(mask)

    def anchors(self):
        return corner.grounded_pairs(self.image, self.charts, self.parts, self.masks, self.projection)

    def test_both_counted_faces_and_the_ground_projection_must_agree(self):
        self.assertEqual({0: (0, 0, -1), 1: (2, 0, -1)}, self.anchors())
        self.masks[0][:self.masks[0].shape[0]//2] = False
        self.assertEqual({}, self.anchors())

    def test_ambiguous_corner_or_incomplete_chart_abstains(self):
        self.parts.append(self.parts[0])
        self.masks.append(self.masks[0])
        self.charts.append(self.charts[0])
        self.assertEqual({}, self.anchors())
        self.charts.pop()
        self.masks.pop()
        self.parts.pop()
        self.charts[0]['pitch']['counts'] = None
        self.assertEqual({}, self.anchors())

    def test_a_separately_detected_step_preserves_the_disputed_cube(self):
        counts = dict(surfaces=[dict(id=c['id'], counts=[4.,5.], quad=c['quad']) for c in self.charts])
        status = dict(P=self.projection, plane_offsets=[dict(id=str(a), axis=a, level=0) for a in (0, 2)])
        def carriers(image, charts, masks, P):
            return [(mask, (axis, -1, -1), 'supported') for mask, axis in zip(masks, (0, 2))]
        def recover(cells):
            for part, mask in zip(self.parts, self.masks):
                part['support'] = dict(size=mask.shape[::-1], runs=row_runs(mask))
            with patch.object(corner, '_carriers', carriers):
                return corner.edits(self.image, self.parts, counts, status, cells)
        cells, _, _ = _blocks(self.masks, carriers(None,None,self.masks,None), self.projection, {})
        added, removed, _ = recover(cells)
        self.assertTrue(added and removed)
        ordered = sorted(cells)
        owners = _frontmost_cells(ordered, self.projection, self.eye, self.masks[0].shape)
        # Model a small visible return belonging to one of the provisional cells.
        # The large rectangular walls retain sufficient overlap for the pair to pass.
        disputed = next(c for c in sorted(removed) if (owners==ordered.index(c)).sum() >= 4)
        hits = np.flatnonzero(owners==ordered.index(disputed))[:6]
        step = np.zeros_like(self.masks[0])
        step.flat[hits] = True
        for mask in self.masks:
            mask[step] = False
        self.masks.append(step)
        self.parts.append(dict(id='step'))
        counts['surfaces'].append(dict(id='step', counts=[None,None], quad=None))
        _, removed, anchors = recover(cells)
        self.assertTrue(anchors)
        self.assertNotIn(disputed, removed)
