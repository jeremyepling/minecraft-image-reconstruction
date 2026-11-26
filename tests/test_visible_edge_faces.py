"""Photographed edge evidence must survive native and coarse visibility checks."""
import unittest

import numpy as np
from PIL import Image

from engine.atlas_count import _prepare
from engine.visible_edge_faces import _near_unclaimed, _support_masks, extend
from engine.visible_placement import _sample_face
from engine.visible_raster import observed_support


def edge_image(size):
    width, height = size
    eye = np.array([.5, .5, -4.])
    basis = np.array([[96., 0., width/2], [0., -96., height/2], [0., 0., 1.]])
    projection = np.column_stack((basis, -basis @ eye))
    y, x = np.indices((height, width))
    point_x = eye[0] + 4*(x+.5-width/2)/96
    point_y = eye[1] - 4*(y+.5-height/2)/96
    face = (point_x >= 1) & (point_x < 2) & (point_y >= 0) & (point_y < 1)
    tiles = np.random.default_rng(113).integers(16, 240, (32, 16, 16, 3), dtype=np.uint8)
    atlas = _prepare(tiles, list(map(str,range(32))), [[str(i)] for i in range(32)], {})
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[face] = tiles[0, np.floor(point_y[face]*16).astype(int), np.floor((point_x[face]-1)*16).astype(int)]
    runs = []
    for y, row in enumerate(face):
        where = np.flatnonzero(row)
        if len(where):
            runs.append([y, int(where[0]), int(where[-1]+1)])
    return Image.fromarray(rgb), projection, atlas, {'size':list(size), 'runs':runs}


class VisibleEdgeFacesTest(unittest.TestCase):
    def test_source_runs_preserve_native_and_grid_masks_at_odd_sizes(self):
        support = {'size':[7,5], 'runs':[[0,0,1],[2,3,6],[4,1,3],[4,6,7]]}
        original = np.zeros((5,7), bool)
        for y,left,right in support['runs']:
            original[y,left:right] = True
        for size in ((7,5), (33,25), (34,26), (35,27), (36,28), (5,3)):
            with self.subTest(size=size):
                shape = tuple((n+1)//4 for n in size[::-1])
                coarse, native = _support_masks(support,size,shape)
                self.assertTrue(np.array_equal(coarse,observed_support(support,shape,size)))
                expected = np.asarray(Image.fromarray(original).resize(size,Image.Resampling.NEAREST))
                self.assertTrue(np.array_equal(native,expected))

    def test_bounding_box_keeps_every_face_with_four_available_rays(self):
        image, projection, _, _ = edge_image((99,67))
        shape = (17,25)
        available = np.random.default_rng(39).random(shape) > .73
        cells = {(x,0,z) for x in range(-4,5) for z in range(-4,8)}
        kept = set(_near_unclaimed(cells,projection,available))
        exact = set()
        for cell in cells:
            for axis in range(3):
                for sign in (-1,1):
                    hit = _sample_face(projection,cell,axis,sign,shape)
                    if hit is not None:
                        box, polygon, *_ = hit
                        if (polygon & available[box]).sum() >= 4:
                            exact.add(cell)
        self.assertTrue(exact)
        self.assertTrue(exact <= kept)
        self.assertGreater(len(cells),len(kept))
        self.assertEqual(_near_unclaimed(cells,projection,np.zeros_like(available)),[])

    def test_extends_matching_face_only_with_image_evidence(self):
        for size in ((97,65), (98,66), (99,67)):
            with self.subTest(size=size):
                image, projection, atlas, support = edge_image(size)
                parts = [{'id':1,'support':support}]
                offsets = [{'id':1,'axis':2,'level':0}]
                cells = {(0,0,0)}
                self.assertEqual(extend(image,parts,projection,cells,offsets,atlas),{(1,0,0)})
                blank = Image.new('RGB',size)
                self.assertFalse(extend(blank,parts,projection,cells,offsets,atlas))
                parts[0]['support'] = {'size':list(size),'runs':[]}
                self.assertFalse(extend(image,parts,projection,cells,offsets,atlas))
                self.assertFalse(extend(image,parts,projection,cells,offsets,None))
