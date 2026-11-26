"""Additional whole-building Java 26.3 photographs for frozen depth decisions."""
import argparse
from functools import partial
from pathlib import Path

from benchmark.camera_fixture import capture, poses


ANGLES = (('low-left', -30, 3.5), ('high-right', 38, 10.2))
PALETTES = {
    'platformshelter': ('copper', ('cut_copper', 'polished_andesite', 'oak_planks')),
    'offsetgate': ('quartz', ('smooth_sandstone', 'quartz_bricks', 'dark_oak_planks')),
}


def platformshelter(b, wall, trim, accent):
    """Two raised platform bays separated by a walk-through aisle under a shared rear rail."""
    for start,length in ((0,8),(11,6)):
        end=start+length
        b.cuboid((start,0,-4),(end,1,2),'platform_base',trim)
        b.cuboid((start,1,-4),(end,2,2),'platform_edge',wall)
        for x in (start,end-1):
            b.cuboid((x,2,-4),(x+1,6,-3),'shelter_post',accent)
        b.cuboid((start+1,2,-3),(end-1,3,-1),'seat',accent)
        b.cuboid((start+1,3,-4),(end-1,4,-3),'seat_back',wall)
        b.cuboid((start-1,6,-5),(end+1,7,1),'shelter_roof',wall)
    b.cuboid((0,5,-4),(17,6,-3),'rear_rail',trim)
    b.cuboid((11,0,2),(15,1,4),'step',accent)


def offsetgate(b, wall, trim, accent):
    """A tall gate and lower side entrance intersect an offset transverse wing."""
    b.cuboid((0,0,-2),(2,8,1),'gate_pier',wall)
    b.cuboid((6,0,-2),(8,8,1),'gate_pier',wall)
    b.cuboid((-1,8,-3),(9,9,2),'high_lintel',trim)
    b.cuboid((2,7,-2),(6,8,0),'inset_lintel',accent)
    b.cuboid((10,0,-4),(12,4,-2),'side_pier',trim)
    b.cuboid((6,4,-4),(13,5,-1),'low_lintel',accent)
    b.cuboid((9,0,-10),(12,4,-8),'rear_wall',wall)
    b.cuboid((10,0,-8),(12,2,-4),'side_sill',wall)
    b.cuboid((9,4,-10),(13,5,-4),'wing_roof',trim)
    b.cuboid((-1,9,-2),(3,10,0),'parapet',accent)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    capture(args.out.resolve(),'light_gray_concrete',shape_groups=(('holdout',platformshelter),('holdout',offsetgate)),
            viewpoints=partial(poses,obliques=ANGLES),palettes_for=lambda s: dict([PALETTES[s.__name__]]))
