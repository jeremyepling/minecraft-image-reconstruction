"""Capture the surface-confirm-v1 buildings in a new clear-noon vanilla Java 26.3 world."""
import argparse
from functools import partial
from itertools import product
from pathlib import Path

from benchmark.camera_fixture import capture, poses
from benchmark.camera_labels import label_capture


FROZEN = Path(__file__).resolve().parent / "data/surface-confirm-v1"
FROZEN_SHA256 = "0a479a8264f2c0f714284928354ff6ff5f9729491709676f07e2033d25e48fc3"


def workshop(b, wall, trim, accent):
    """A terraced gable, a cut loading bay and an offset lower annex."""
    b.cuboid((0, 0, -6), (12, 6, 1), "workshop", wall)
    for xyz in product(range(2, 6), range(0, 4), range(-2, 1)):
        del b.cells[xyz]
    b.cuboid((2, 0, -3), (6, 4, -2), "loading_bay_back", accent)
    for tier in range(5):
        b.cuboid((tier-1, 6+tier, -7), (13-tier, 7+tier, 2), "gable_roof", trim)
    b.cuboid((12, 0, -8), (17, 4, -1), "annex", wall)
    b.cuboid((12, 4, -8), (18, 5, 0), "annex_eave", accent)


def crossing(b, wall, trim, accent):
    """An elevated crossing between offset towers; the eastern half has a recessed back."""
    b.cuboid((0, 0, -5), (4, 8, 1), "west_tower", wall)
    b.cuboid((12, 0, -6), (16, 10, 0), "east_tower", trim)
    b.cuboid((3, 5, -4), (15, 7, 3), "bridge_and_soffit", accent)
    b.cuboid((3, 7, 2), (15, 8, 3), "balcony_parapet", wall)
    b.cuboid((9, 0, -5), (12, 5, -4), "passage_back", wall)
    for step in range(3):
        b.cuboid((9, 0, 2-step), (12, 1+step, 3-step), "entry_steps", trim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New capture directory")
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--minecraft-dir", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    angles = (("low-left", -34, 2.8), ("high-right", 23, 8.8))
    capture(out, "light_gray_concrete", args.jar, args.minecraft_dir,
            shape_groups=(("holdout", workshop), ("holdout", crossing)),
            viewpoints=partial(poses, obliques=angles))
    label_capture(out)


if __name__ == "__main__":
    main()
