"""Sealed four-view native roof-height transfer fixture for Java 26.3."""
import argparse
from functools import partial
from pathlib import Path

from benchmark.camera_fixture import capture, poses

ANGLES = (('low-left', -28, 4.1), ('high-right', 34, 11.5))
PALETTES = {
    'lowawning': ('masonry', ('stone_bricks', 'polished_andesite', 'chiseled_stone_bricks')),
    'highwalkroof': ('timber', ('spruce_planks', 'stripped_oak_log', 'oak_planks')),
}


def lowawning(b, roof, post, accent):
    for x in (0, 6):
        for z in (0, -3):
            b.cuboid((x, 0, z), (x + 1, 3, z + 1), 'pier', post)
    b.cuboid((-1, 3, -4), (8, 4, 2), 'roof', roof)
    b.cuboid((-1, 3, 1), (8, 4, 2), 'roof_edge', accent)


def highwalkroof(b, roof, post, accent):
    for x, z in ((0, 0), (5, 0), (10, 0), (0, -5), (10, -5)):
        b.cuboid((x, 0, z), (x + 1, 8, z + 1), 'post', post)
    b.cuboid((-1, 8, -6), (13, 9, 1), 'roof', roof)
    b.cuboid((-1, 8, 1), (7, 9, 3), 'projecting_roof', accent)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    capture(args.out.resolve(), 'smooth_stone',
            shape_groups=(('holdout', lowawning), ('holdout', highwalkroof)),
            viewpoints=partial(poses, obliques=ANGLES),
            palettes_for=lambda shape: dict([PALETTES[shape.__name__]]))
