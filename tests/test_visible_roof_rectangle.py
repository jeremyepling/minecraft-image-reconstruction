from unittest import TestCase
from unittest.mock import patch

from engine import visible_roof_rectangle as roof


class VisibleRoofRectangleTests(TestCase):
    def test_rim_legs_need_a_common_corner_contiguous_in_both_directions(self):
        first={(0,3,0),(0,3,1),(0,3,3)}
        second={(0,3,0),(1,3,0),(2,3,0)}
        cells,n=roof.rects({0:first,2:second},(-1,6,-1),first|second)
        self.assertEqual(n,1)
        self.assertEqual(cells,{(1,3,1),(2,3,1)})
        self.assertEqual(roof.rects({0:first,2:second-{(0,3,0)}},(-1,6,-1),first|second)[0],set())

    def test_contiguous_requires_both_shared_patch_and_new_pixels(self):
        with patch.object(roof,'recover',return_value=(set(),{})) as recover:
            roof.contiguous(None,[],None,{},set(),atlas_path='custom')
            self.assertEqual(recover.call_args.kwargs,dict(texture=False,unclaimed=True,shared_patch=True,atlas_path='custom'))
