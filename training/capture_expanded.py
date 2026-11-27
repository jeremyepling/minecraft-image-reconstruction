"""Capture new whole-structure Minecraft 26.3 datasets in an isolated client."""
import argparse
import math
import subprocess
import sys
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

from benchmark.camera_fixture import GROUND, VERSION, stage
from engine.atlas import discover
from runtime.mode import require_game_runtime


FULL_ANGLES = (("low-left", -24, .12), ("high-right", 32, .55), ("rear-oblique", 151, .63))
DECOR_ANGLES = (("low-left", -22, .27), ("high-right", 30, .70))
RESOLUTION = (1440, 900)


def framed_views(building, shape, lo, hi, *, angles):
    """Find a camera that frames the entire bounding box with room to spare."""
    center = (np.array(lo) + hi) / 2
    target = np.array(building.xyz(center))
    corners = np.array(list(product(*zip(lo, hi)))) - center
    width, height = RESOLUTION
    tangent = math.tan(math.radians(55 / 2))
    for angle, azimuth, relative_height in angles:
        radians = math.radians(azimuth)
        horizontal = np.array([math.sin(radians), 0, math.cos(radians)])
        eye_height = max(3.0, (hi[1] - lo[1]) * relative_height)
        elevation = lo[1] + eye_height - center[1]
        distance = max(8.0, float(np.max(hi - np.array(lo))) * .70)
        while True:
            offset = horizontal * distance + [0, elevation, 0]
            forward = -offset / np.linalg.norm(offset)
            right = np.cross(forward, [0, 1, 0])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            depth = (corners - offset) @ forward
            if (depth.min() > 0
                    and np.max(np.abs(corners @ right) / depth) < tangent * width / height * .79
                    and np.max(np.abs(corners @ up) / depth) < tangent * .79):
                break
            distance *= 1.08
        eye = target + offset
        yield dict(angle=angle, condition="whole-structure", target=target.tolist(),
                   x=float(eye[0]), z=float(eye[2]), feetY=GROUND + 1 + lo[1] + eye_height - 1.62)


def capture(root, groups, palettes_for, *, angles, jar=None, render_distance=12):
    require_game_runtime("expanded dataset capture")
    jar = (jar or discover(VERSION)).resolve()
    stage(root, jar, "light_gray_concrete", shape_groups=groups, palettes_for=palettes_for,
          viewpoints=partial(framed_views, angles=angles), sparse=True, page_spacing=384)
    subprocess.run([sys.executable, "-m", "runtime.capture", "capture", "--version", VERSION,
                    "--run", str(root), "--jar", str(jar), "--resolution", *map(str, RESOLUTION),
                    "--render-distance", str(render_distance)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("full", "decorative"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jar", type=Path)
    args = parser.parse_args()
    if args.dataset == "full":
        from training.large_layouts import GROUPS, palettes_for

        angles, render_distance = FULL_ANGLES, 12
    else:
        from training.decorative_layouts import GROUPS, palettes_for

        angles, render_distance = DECOR_ANGLES, 8
    capture(args.out.resolve(), GROUPS, palettes_for, angles=angles, jar=args.jar,
            render_distance=render_distance)


if __name__ == "__main__":
    main()
