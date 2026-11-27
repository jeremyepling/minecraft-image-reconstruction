import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from engine.atlas_count import _prepare
from engine.placed_materials import FALLBACK, _stabilize, assign


class PlacedMaterials(unittest.TestCase):
    def test_only_frontmost_source_faces_get_matched(self):
        colors = np.array([[200,22,17],[20,170,195]], dtype=np.uint8)
        tiles = np.broadcast_to(colors[:,None,None,:], (2,16,16,3))
        atlas = _prepare(tiles, ['red_concrete','cyan_concrete'],
                         [['minecraft:red_concrete'],['minecraft:cyan_concrete']], {})
        camera = {'P':[[64,0,32,-32],[0,-64,32,32],[0,0,1,0]]}
        source = Image.new('RGB',(64,64),(157,17,14))
        cells = [(0,0,4),(0,0,5),(100,0,4)]
        ids, sources = assign(source,camera,cells,atlas)
        self.assertEqual(ids, {cells[0]:'red_concrete',cells[1]:FALLBACK,cells[2]:FALLBACK})
        self.assertEqual(sources, {cells[0]:'source-atlas',cells[1]:'fallback',cells[2]:'fallback'})
        self.assertEqual(assign(source,camera,reversed(cells),atlas), (ids,sources))
        for missing_camera, missing_atlas in ((None,atlas),(camera,None),({'P':np.zeros((3,4))},atlas)):
            both = assign(source,missing_camera,cells,missing_atlas)
            self.assertEqual(both,(dict.fromkeys(cells,FALLBACK),dict.fromkeys(cells,'fallback')))

    def test_gravity_uses_next_ranked_stable_id_only_at_unsupported_positions(self):
        cells = [(0,0,0),(0,1,0),(2,3,0)]
        ids = dict(zip(cells,['stone','sand','red_concrete_powder']))
        evidence = dict.fromkeys(cells,'source-atlas')
        catalog = SimpleNamespace(blocks=['red_concrete_powder','sand','stone','pink_concrete'])
        costs = np.array([[8,5,0,4],[7,0,1,9],[0,1,8,3]])
        _stabilize(ids,evidence,costs,cells,catalog)
        self.assertEqual(list(ids.values()), ['stone','sand','pink_concrete'])
        self.assertEqual(evidence[cells[1]],'source-atlas')
        self.assertEqual(evidence[cells[2]],'source-atlas:unsupported_gravity_block:red_concrete_powder')
