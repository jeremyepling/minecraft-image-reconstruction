import unittest

import numpy as np

from engine.junction_depth import JunctionDepth


def row(level, **values):
    return dict(dict(level=level, atlas=.1, edge=.1, boundary=.3,
                     distinct_votes=0, tiles=0, scene_delta=0), **values)


class JunctionDepthTest(unittest.TestCase):
    def test_contacts_on_two_independent_walls_localize_height(self):
        target = np.zeros((16,16), bool)
        target[2:13,2:13] = True
        wall_x = np.zeros_like(target)
        wall_z = np.zeros_like(target)
        wall_x[13,2:13] = True
        wall_z[2:13,13] = True
        eye = np.array([0.,5.,0.])
        rays = np.broadcast_to([1.,-.5,1.], target.shape+(3,))
        finder = JunctionDepth((64,64), [])
        finder.anchors = {}
        finder.walls = [(wall_x,0,4), (wall_z,2,4)]
        self.assertEqual(finder.levels(target,None,eye,rays,1), {3})
        def choose():
            return finder.choose({'id':'roof'},target,None,eye,rays,1,2,4,
                                 [row(2),row(3)],np.zeros(12),chosen_before=2)
        self.assertEqual(choose(), 3)
        finder.walls.pop()
        finder.levels(target,None,eye,rays,1)
        self.assertEqual(choose(), 2)

    def test_contact_does_not_cross_the_viewport_border(self):
        wall = np.zeros((16,16), bool)
        wall[:,-1] = True
        target = np.zeros_like(wall)
        target[:,0] = True
        finder = JunctionDepth((64,64), [])
        finder.walls = [(wall,0,4)]
        rays = np.broadcast_to([1.,-.5,1.], target.shape+(3,))
        self.assertFalse(finder.crease_levels(target,np.array([0,5,0]),rays))


if __name__ == '__main__':
    unittest.main()
