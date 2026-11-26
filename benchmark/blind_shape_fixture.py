"""Create four sealed native photographs of two held-out full-cube structures."""
import argparse
from functools import partial
from pathlib import Path

from benchmark.camera_fixture import capture, poses

FLOOR = 'light_gray_concrete'
ANGLES = (('low-left', -32, 3.1), ('high-right', 36, 10.1))
PALETTES = {
    'switchback': ('brick', ('bricks', 'polished_andesite', 'birch_planks')),
    'loadingdock': ('timber', ('spruce_planks', 'stone_bricks', 'oxidized_copper')),
}


def switchback(b, wall, trim, accent):
    """Two parallel flights around an open well with a cantilevered landing."""
    for step in range(4):
        b.cuboid((step*2,0,1), (step*2+2,step+1,4), 'first_flight',wall)
        b.cuboid((6-step*2,4+step,-4), (8-step*2,5+step,-1), 'upper_flight',trim)
    b.cuboid((8,3,-4),(12,4,4),'landing',accent)
    b.cuboid((9,0,-4),(11,3,-2),'landing_pier',wall)
    b.cuboid((0,0,-4),(1,7,-2),'upper_pier',wall)
    b.cuboid((-2,7,-5),(2,8,0),'lookout',accent)
    b.cuboid((-2,8,-5),(-1,10,-4),'lookout_post',trim)
    b.cuboid((1,8,-5),(2,10,-4),'lookout_post',trim)
    b.cuboid((-2,10,-5),(2,11,-4),'lookout_beam',wall)


def loadingdock(b, wall, trim, accent):
    """Two different recessed facade depths behind an offset deck and overhead rail."""
    b.cuboid((0,0,-7),(15,7,-5),'rear',wall)
    b.cuboid((0,0,-5),(3,7,0),'store',wall)
    b.cuboid((11,0,-5),(15,7,-1),'store',wall)
    b.cuboid((0,6,-8),(15,7,1),'roof',trim)
    b.cuboid((3,0,-5),(11,3,-2),'inset',accent)
    b.cuboid((3,1,-2),(14,2,4),'dock',trim)
    for x in (3,8,13):
        b.cuboid((x,0,2),(x+1,1,4),'dock_foot',wall)
    for x in (5,13):
        b.cuboid((x,2,2),(x+1,7,3),'gantry_post',accent)
    b.cuboid((4,7,2),(15,8,3),'gantry_beam',wall)
    b.cuboid((7,8,2),(10,9,3),'trolley',trim)


def acquire(root):
    shapes = (('holdout',switchback),('holdout',loadingdock))
    capture(root, FLOOR, shape_groups=shapes, viewpoints=partial(poses,obliques=ANGLES),
            palettes_for=lambda shape: dict([PALETTES[shape.__name__]]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    acquire(parser.parse_args().out.resolve())
