"""Stage, capture and label paired vanilla Java 26.3 Floating Front structures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.camera_fixture import VERSION, write
from calibration.world import PACK, create_world
from engine.atlas import digest, discover
from training.front_layouts import design, camera
from training.floating_front_release import CLOCK

LIGHTS = tuple((name, CLOCK[name]) for name in ("noon", "morning", "afternoon"))


def stage(root, first, last, jar):
    with ZipFile(jar) as archive:
        version = json.loads(archive.read("version.json"))
    if version["id"] != VERSION:
        raise ValueError(f"Expected Java {VERSION}")
    root.mkdir(parents=True, exist_ok=False)
    world = root / "fixture/saves/Palette Lab"
    create_world(world, version, {"minecraft:random_tick_speed": 0}, "Floating Front 100", floor=None)
    (world / "datapacks").mkdir()
    views, blocks = [], []
    with ZipFile(world / "datapacks" / PACK, "w", ZIP_DEFLATED) as pack:
        fmt = [version["pack_version"][key] for key in ("data_major", "data_minor")]
        pack.writestr("pack.mcmeta", json.dumps(dict(pack=dict(description="Floating Front 100",
                                                                min_format=fmt, max_format=fmt))))
        for number in range(first, last + 1):
            info, base, decor = design(number)
            actual_size = [max(axis) - min(axis) + 1 for axis in zip(*base)]
            if (actual_size != info["dimensions"] or (info["size"] == "large" and actual_size[1] != 50)
                    or base.keys() & decor.keys()):
                raise ValueError(f"Invalid dimensions or overwritten cube in {info['layout']}")

            def at_world(cells):
                return {(x + number * 384, y + 145, z): block for (x, y, z), block in cells.items()}
            base, decor = at_world(base), at_world(decor)
            bounds = [min(point[i] for point in base) for i in range(3)]
            # The crane's projecting fence otherwise leaves the bare boom at 74.375% of frame.
            frame_fraction = .94 if info["layout"] == "front-067-crane" else .90
            pose = camera(base.keys() | decor.keys(), info["angle_deg"], info["elevation_deg"],
                          frame_fraction=frame_fraction)
            for condition in ("full", "decorative"):
                page = 2 * number + int(condition == "decorative")
                extra = decor if condition == "decorative" else {p: "air" for p in decor}
                scene = base | extra
                for (x, y, z), state in scene.items():
                    blocks.append(dict(page=page, x=x, y=y, z=z, block_id=state.split("[")[0], expected=state))
                place = base if condition == "full" else decor
                commands = [f"setblock {x} {y} {z} minecraft:{state} strict" for (x,y,z), state in place.items()]
                pack.writestr(f"data/palette_lab/function/page_{page}.mcfunction", "\n".join(commands) + "\n")
                for lighting, ticks in (LIGHTS if condition == "full" else LIGHTS[:1]):
                    layout = info["layout"]
                    views.append(dict(**info, **pose, id=f"{layout}-{condition}-{lighting}",
                                      parent_id=f"{layout}-full-noon", page=page, number=number,
                                      origin_world=bounds, cloudless=True, angle="front", lighting=lighting,
                                      clock_ticks=ticks, condition=condition))
    write(root / "layout.json", dict(version=VERSION, client_sha256=digest(jar), blocks=blocks))
    write(root / "views.json", views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=100)
    parser.add_argument("--jar", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--capture-only", action="store_true")
    modes.add_argument("--labels-only", action="store_true")
    modes.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.first <= args.last <= 100:
        parser.error("Expected a range between 1 and 100 inclusive")
    jar = (args.jar or discover(VERSION)).resolve()
    root = args.out.resolve()
    if not args.labels_only:
        stage(root, args.first, args.last, jar)
        if args.stage_only:
            return
        subprocess.run([sys.executable, "-m", "runtime.capture", "capture", "--version", VERSION,
                        "--jar", str(jar), "--run", str(root)], check=True)
    if not args.capture_only:
        from training.front_labels import finalize
        finalize(root, jar)


if __name__ == "__main__":
    main()
