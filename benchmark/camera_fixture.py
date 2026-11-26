"""Build and capture the shared Java 26.3 exterior fixture. All truth files are for evaluation."""
import argparse
import json
import math
import subprocess
import sys
from itertools import product
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.fixtures import Building
from calibration.world import GROUND, PACK, create_world
from engine.atlas import digest, discover
from runtime.mode import require_game_runtime


VERSION = "26.3"
FLOORS = ("smooth_stone", "light_gray_concrete")
PALETTES = {
    "masonry": ("stone_bricks", "tuff", "polished_andesite"),
    "plaster": ("white_concrete", "cobblestone", "spruce_planks"),
}


def wing(b, wall, trim, accent):
    """An uninterrupted wall with a return and a lower side wing."""
    b.cuboid((0, 0, -3), (13, 8, 1), "wall", wall)
    b.cuboid((1, 0, -8), (5, 6, -3), "return", trim)
    b.cuboid((13, 0, -4), (17, 4, 0), "wing", accent)


def shopfront(b, wall, trim, accent):
    """Alternating shallow and deep openings, with an awning above two piers."""
    b.cuboid((0, 0, -4), (15, 7, 1), "facade", wall)
    for x, depth in ((2, 1), (7, 3), (11, 2)):
        for i, j, k in product(range(x, x + 2), range(1, 5), range(1 - depth, 1)):
            del b.cells[i, j, k]
        b.cuboid((x, 1, -depth), (x + 2, 5, 1 - depth), "recess_back", trim)
    b.cuboid((5, 5, 1), (15, 6, 4), "awning", accent)
    for x in (5, 13):
        b.cuboid((x, 0, 2), (x + 1, 5, 3), "pier", trim)


def courtyard(b, wall, trim, accent):
    """Two wings of different heights enclosing a deep open corner."""
    b.cuboid((0, 0, -6), (16, 8, -4), "rear_wall", wall)
    b.cuboid((0, 0, -4), (4, 6, 2), "near_wing", trim)
    b.cuboid((12, 0, -4), (16, 5, 0), "far_wing", accent)
    b.cuboid((0, 6, -6), (5, 7, 2), "coping", wall)


def portico(b, wall, trim, accent):
    """Three open front bays, a projecting roof, and a small enclosed end bay."""
    b.cuboid((0, 0, -5), (16, 6, -4), "rear_wall", wall)
    b.cuboid((12, 0, -4), (16, 6, 0), "end_bay", wall)
    for x in (0, 4, 8, 12):
        b.cuboid((x, 0, 1), (x + 1, 5, 2), "pillar", trim)
    b.cuboid((-1, 5, -4), (17, 6, 3), "canopy", accent)
    b.cuboid((2, 6, -5), (8, 9, -2), "upper_wall", wall)


LAYOUTS = (("dev", wing), ("dev", shopfront), ("holdout", courtyard), ("holdout", portico))


def write(path, data):
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf8")


def poses(b, shape, lo, hi, *, obliques=(("low-left", -27, 3.1), ("high-right", 34, 8.3))):
    center = b.xyz([(a + z) / 2 for a, z in zip(lo, hi)])
    for angle, azimuth, eye_height in obliques:
        distance = 1.32 * (hi[0] - lo[0]) + .4 * (hi[2] - lo[2])
        az = math.radians(azimuth)
        yield dict(angle=angle, condition="ground-visible", target=center,
                   x=center[0] + distance * math.sin(az), z=center[2] + distance * math.cos(az),
                   feetY=GROUND + 1 + eye_height - 1.62)
    if shape in (wing, courtyard):
        # Inside the courtyard aperture both side wings are outside this camera crop.
        wall_z = 1 if shape is wing else -4
        x = 7 if shape is wing else 8
        yield dict(angle="close-front", condition="limited-ground", target=b.xyz([x, 4, wall_z]),
                   x=b.origin + x, z=wall_z + 3.9, feetY=GROUND + 1 + 4 - 1.62)


def stage(root, jar, floor="smooth_stone", *, shape_groups=LAYOUTS, viewpoints=poses, palettes_for=None):
    if floor not in FLOORS:
        raise ValueError(f"Unknown fixture floor: {floor}")
    with ZipFile(jar) as archive:
        meta = json.loads(archive.read("version.json"))
    if meta["id"] != VERSION:
        raise ValueError(f"Expected Java {VERSION}, found {meta['id']}")
    root.mkdir(parents=True, exist_ok=False)
    world = root / "fixture/saves/Palette Lab"
    create_world(world, meta, {"minecraft:random_tick_speed": 0}, level_name="Exterior fixture", floor=floor)
    (world / "datapacks").mkdir()
    blocks, views, layouts = [], [], []
    with ZipFile(world / "datapacks" / PACK, "w", ZIP_DEFLATED) as pack:
        fmt = [meta["pack_version"][k] for k in ("data_major", "data_minor")]
        pack.writestr("pack.mcmeta", json.dumps(dict(pack=dict(description="Exterior fixture",
                                                              min_format=fmt, max_format=fmt))))
        for split, shape in shape_groups:
            palettes = palettes_for(shape) if palettes_for else PALETTES
            for palette_name, materials in palettes.items():
                page = len(layouts)
                internal_split = {"holdout": "test", "validation": "dev"}.get(split, split)
                b = Building(page, internal_split, shape.__name__, {})
                b.origin = page * 256  # Other pages are outside the private client's render distance.
                shape(b, *materials)
                lo, hi = b.finish()
                common = dict(page=page, split=split, layout=shape.__name__, palette=palette_name)
                layouts.append(dict(**common, bounds=b.data["bounds"], planes=b.data["planes"]))
                commands = []
                # Read untouched worldgen surface blocks below the approaches to each view.
                for x, z in ((-4, 18), (8, 12), (21, 16)):
                    blocks.append(dict(page=page, x=b.origin + x, y=GROUND, z=z,
                                       block_id=floor, role="floor_probe"))
                for pos in product(*(range(a, z) for a, z in zip(lo, hi))):
                    material, role = b.cells.get(pos, ("air", "open"))
                    x, y, z = b.xyz(pos)
                    blocks.append(dict(page=page, x=x, y=y, z=z, block_id=material, role=role))
                    if material != "air":
                        commands.append(f"setblock {x} {y} {z} minecraft:{material} strict")
                pack.writestr(f"data/palette_lab/function/page_{page}.mcfunction", "\n".join(commands) + "\n")
                for view in viewpoints(b, shape, lo, hi):
                    view_id = f"{split}-{shape.__name__}-{palette_name}-{view['angle']}"
                    views.append(dict(**common, id=view_id, **view))
    write(root / "layout.json", dict(version=VERSION, client_sha256=digest(jar), ground_y=GROUND, floor=floor,
                                     blocks=blocks))
    write(root / "buildings.json", layouts)
    write(root / "views.json", views)


def capture(root, floor, jar=None, minecraft_dir=None, **stage_options):
    require_game_runtime('fixture capture')
    if jar is None and minecraft_dir:
        jar = minecraft_dir / "versions" / VERSION / f"{VERSION}.jar"
    jar = (jar or discover(VERSION)).resolve()
    stage(root, jar, floor, **stage_options)
    command = [sys.executable, "-m", "runtime.capture", "capture", "--version", VERSION,
               "--run", str(root), "--jar", str(jar)]
    if minecraft_dir:
        command.extend(["--minecraft-dir", str(minecraft_dir)])
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--minecraft-dir", type=Path)
    parser.add_argument("--floor", choices=FLOORS, default="smooth_stone", help="Superflat surface block")
    parser.add_argument("--labels-only", action="store_true", help="Verify and label an existing capture")
    parser.add_argument("--paired-with", type=Path, help="Audit this capture against a matched reference fixture")
    args = parser.parse_args()
    root = args.out.resolve()
    if not args.labels_only:
        capture(root, args.floor, args.jar, args.minecraft_dir)
    from benchmark.camera_labels import label_capture, verify_pair
    label_capture(root)
    if args.paired_with:
        verify_pair(args.paired_with, root)


if __name__ == "__main__":
    main()
