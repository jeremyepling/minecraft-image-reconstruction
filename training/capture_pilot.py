"""Capture the 96 pilot views with the same Java driver used by the frozen regression suite."""
import argparse
import json
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_fixture import capture, poses, write
from benchmark.camera_labels import label_capture
from engine.atlas import digest


# Palette entries select a set of wall, trim and accent materials for each family.
PALETTES = {
    "earth": [
        ("bricks", "smooth_sandstone", "oak_planks"),
        ("terracotta", "mud_bricks", "cut_sandstone"),
        ("granite", "polished_granite", "jungle_planks"),
        ("sandstone", "red_sandstone", "smooth_quartz"),
    ],
    "cool": [
        ("deepslate_tiles", "light_gray_concrete", "prismarine_bricks"),
        ("polished_andesite", "stone_bricks", "spruce_planks"),
        ("blue_terracotta", "calcite", "dark_prismarine"),
        ("cobblestone", "polished_diorite", "warped_planks"),
    ],
}
GROUPS = [
    ("train", "arcade"), ("train", "steppedhall"), ("train", "sawtooth"),
    ("train", "terrace"), ("train", "watchpost"), ("train", "arcbridge"),
    ("validation", "cloister"), ("test", "twinspire"),
]
ANGLES = (("low-left", -32, 3.1), ("high-right", 36, 10.1))
FLOOR = "light_gray_concrete"


def layout(family, index):
    def build(b, wall, trim, accent):
        w, d, h = 12 + index * 3, 6 + index, 5 + index
        box = b.cuboid

        def cut(lo, hi):
            for pos in product(*(range(a, z) for a, z in zip(lo, hi))):
                b.cells.pop(pos, None)

        if family == "arcade":
            box((0, 0, -d), (w, h, -d + 1), "back", wall)
            for x in range(0, w, 4):
                box((x, 0, 1), (x + 1, h, 3), "column", trim)
            box((-1, h, -d - 1), (w + 1, h + 1, 4), "canopy", accent)
            box((w - 3, 0, -d), (w, h, 1), "end", wall)
        elif family == "steppedhall":
            box((0, 0, -d), (w, h, 1), "hall", wall)
            cut((2, 0, -1), (4, h - 1, 2))
            for step in range(3 + index):
                box((step, h + step, -d - 1), (w - step, h + step + 1, 2), "gable", trim)
            box((w - 2, 1, 1), (w + 1, h - 1, 3), "bay", accent)
        elif family == "sawtooth":
            box((0, 0, -d), (w, h, 1), "factory", wall)
            for x in range(w):
                box((x, h, -d - 1), (x + 1, h + 1 + x % 4, 2), "shed_roof", trim)
            for x in range(1, w - 2, 4):
                cut((x, 1, 0), (x + 2, h - 1, 1))
                box((x, 1, -1), (x + 2, h - 1, 0), "window_back", accent)
        elif family == "terrace":
            for level in range(3):
                box((2 * level, level * 3, -d), (w - level, level * 3 + 3, 2 - level * 2), "tier", wall)
                box((2 * level, level * 3 + 2, 1 - level * 2), (w - level, level * 3 + 3, 2 - level * 2), "band", trim)
            box((0, 0, 2), (w // 2, 1, 4), "forecourt", accent)
        elif family == "watchpost":
            box((2, 0, -d), (w - 3, h + 2, -1), "base", wall)
            box((0, h + 2, -d - 2), (w, h + 3, 1), "platform", trim)
            for x, z in product((1, w - 2), (-d - 1, -1)):
                box((x, h + 3, z), (x + 1, h + 5, z + 1), "post", accent)
            box((-1, h + 5, -d - 2), (w + 1, h + 6, 2), "roof", wall)
        elif family == "arcbridge":
            box((0, 0, -d), (w, h, 1), "span", wall)
            cut((2, 0, -d), (w - 2, h - 3, 1))
            cut((3, 0, -d), (w - 3, h - 2, 1))
            cut((4, 0, -d), (w - 4, h - 1, 1))
            box((-1, h, -d - 1), (w + 1, h + 1, 2), "deck", trim)
            for x in range(0, w, 2):
                box((x, h + 1, 1), (x + 1, h + 2, 2), "railing", accent)
        elif family == "cloister":
            box((0, 0, -d), (w, h, 1), "ring", wall)
            cut((2, 0, -d + 2), (w - 2, h, -1))
            cut((w // 2 - 1, 0, -1), (w // 2 + 1, h - 1, 1))
            for x in (0, w - 3):
                box((x, h, -d), (x + 3, h + 2, 1), "upper_wing", trim)
            box((0, h, -d - 1), (w, h + 1, -d + 1), "rear_eave", accent)
        elif family == "twinspire":
            for x, height in ((0, h + 2), (w - 4, h + 4)):
                box((x, 0, -d), (x + 4, height, 1), "tower", wall)
                box((x - 1, height, -d - 1), (x + 5, height + 1, 2), "cornice", trim)
                box((x + 1, height + 1, -d + 1), (x + 3, height + 3, 0), "lantern", accent)
            box((3, h - 2, -d + 1), (w - 3, h, -1), "crossing", trim)
            box((3, 0, -d), (w - 3, 2, -d + 1), "low_wall", accent)

    build.__name__ = f"{family}{index+1}"
    return build


def palette_for_layout(name):
    family, size = name[:-1], int(name[-1]) - 1
    family_index = [family_name for _, family_name in GROUPS].index(family)
    return {label: variants[(family_index + size) % len(variants)] for label, variants in PALETTES.items()}


def verify_palette_pairs(root, manifest):
    """Check that projected, unobstructed face centers really change color in paired native images."""
    pairs = {}
    for frame in manifest["frames"]:
        pairs.setdefault((frame["layout"], frame["angle"]), {})[frame["palette"]] = frame
    rows = []
    for (layout_name, angle), pair in sorted(pairs.items()):
        earth, cool = pair["earth"], pair["cool"]
        faces = json.loads((root / earth["labels"]).read_text())["faces"]
        centers = [np.mean(face["image_corners"], axis=0) for face in faces
                   if face["visible_fraction"] == 1 and face["visible_pixels"] >= 64]
        if len(centers) < 10:
            raise ValueError(f"Too few interior samples on {layout_name}/{angle}")
        x, y = np.floor(centers).astype(int).T
        colors = []
        for frame in (earth, cool):
            with Image.open(root / frame["image"]) as image:
                colors.append(np.asarray(image.convert("RGB"))[y, x].astype(int))
        different = np.max(abs(colors[0] - colors[1]), axis=1) >= 8
        fraction = float(np.mean(different))
        rows.append(dict(layout=layout_name, angle=angle, samples=len(centers),
                         visibly_different=int(different.sum()), fraction=fraction))
        if fraction < 0.85:
            raise ValueError(f"Only {fraction:.1%} of {layout_name}/{angle} face centers differ across palettes")
    return dict(method="Native RGB at projected centers of fully visible faces with at least 64 pixels, paired by layout and angle. Max channel difference >= 8/255.", pairs=rows)


def finalize(root):
    label_capture(root)
    manifest = json.loads((root / "index.json").read_text())
    sites = json.loads((root / "buildings.json").read_text())
    for frame in manifest["frames"]:
        site = next(s for s in sites if (s["layout"], s["palette"]) == (frame["layout"], frame["palette"]))
        family = frame["layout"][:-1]
        frame.update(version=manifest["version"], client_sha256=manifest["client_sha256"],
                     ground_y=manifest["ground_y"], family=family,
                     materials=palette_for_layout(frame["layout"])[frame["palette"]],
                     origin_world=site["bounds"][0], readback=frame["states"])
        frame["source_sha256"] = {key: digest(root/frame[key]) for key in ("image","pose","readback")}
    if len(manifest["frames"]) != 96 or len({r["image_sha256"] for r in manifest["frames"]}) != 96:
        raise ValueError("Expected 96 distinct captured screenshots")
    manifest["palette_audit"] = verify_palette_pairs(root, manifest)
    write(root / "index.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if not args.labels_only:
        shapes = [(split, layout(family, i)) for split, family in GROUPS for i in range(3)]
        capture(root, FLOOR, shape_groups=shapes, viewpoints=partial(poses, obliques=ANGLES),
                palettes_for=lambda shape: palette_for_layout(shape.__name__))
    finalize(root)


if __name__ == "__main__":
    main()
