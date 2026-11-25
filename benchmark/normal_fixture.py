"""Capture the normal-confirm-v1 buildings and optionally archive their playable save."""
import argparse
import json
from functools import partial
from pathlib import Path

from benchmark.camera_fixture import capture, poses
from benchmark.camera_labels import label_capture
from benchmark.world_archives import audit, digest, load, pack


FROZEN = Path(__file__).resolve().parent / "data/normal-confirm-v1"
FROZEN_SHA256 = "61bdf3a89a505a8be3757efe480185bec9c1888a1c72ffb69ef0093623e27dda"
FLOOR = "light_gray_concrete"
ANGLES = (("low-left", -23, 3.4), ("high-right", 38, 9.5))


def registry(b, wall, trim, accent):
    """Two enclosed interlocking volumes on shifted axes, one cantilevered over its footing."""
    b.cuboid((0, 0, -6), (8, 7, 2), "west_hall", wall)
    b.cuboid((0, 7, -6), (8, 8, 2), "west_coping", trim)
    b.cuboid((7, 0, -9), (16, 3, -2), "east_footing", accent)
    b.cuboid((6, 3, -10), (17, 10, -1), "east_hall", wall)
    b.cuboid((6, 10, -10), (17, 11, -1), "east_coping", trim)
    b.cuboid((3, 0, 2), (7, 4, 4), "vestibule", accent)


def galleries(b, wall, trim, accent):
    """Stacked projecting galleries with staggered supports, open spans and a backed inner bay."""
    for x in (0, 6, 13):
        b.cuboid((x, 0, 0), (x + 2, 4, 2), "lower_pier", trim)
        b.cuboid((x, 0, -6), (x + 2, 4, -4), "rear_pier", trim)
    b.cuboid((0, 0, -6), (8, 4, -5), "lower_bay_back", wall)
    b.cuboid((-1, 4, -7), (16, 5, 3), "main_gallery", accent)
    for x in (2, 10):
        b.cuboid((x, 5, -2), (x + 2, 8, 0), "upper_pier", wall)
    b.cuboid((2, 5, -6), (12, 8, -5), "upper_bay_back", trim)
    b.cuboid((1, 8, -6), (13, 9, 1), "upper_gallery", accent)
    b.cuboid((1, 9, -6), (13, 10, -5), "rear_parapet", wall)
    b.cuboid((12, 9, -5), (13, 10, 1), "side_parapet", wall)


def archive_world(root, destination):
    title = "Normal confirmation v1"
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "light-gray-concrete.zip"
    pack(root / "client/saves/Palette Lab", archive, title)
    world = audit(archive, title, root, FLOOR)
    views = load(root / "views.json")
    sites = []
    for site in load(root / "buildings.json"):
        stations = []
        for view in (v for v in views if v["page"] == site["page"]):
            pose = load(root / "captures" / (view["id"] + ".json"))
            yaw, pitch = pose["yaw_pitch_deg"]
            stations.append(dict(image=view["id"], teleport=(
                f"/tp @s {view['x']:.5f} {view['feetY']:.5f} {view['z']:.5f} {yaw:.5f} {pitch:.5f}")))
        sites.append(dict(layout=site["layout"], palette=site["palette"], split=site["split"],
                          bounds_xyz=site["bounds"], stations=stations))
    manifest = dict(schema=1, fixture="normal-confirm-v1", fixture_index_sha256=digest(root / "index.json"),
                    game="Minecraft Java Edition", version="26.3", worlds=[world], sites=sites,
                    install=f"Extract the ZIP into the Java game directory's saves folder. Open '{title}' in Singleplayer.",
                    bounds="World XYZ, lower inclusive and upper exclusive.")
    (destination / "world-index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New capture directory")
    parser.add_argument("--world-out", type=Path, help="Also package and verify the playable save in a new directory")
    parser.add_argument("--jar", type=Path)
    parser.add_argument("--minecraft-dir", type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    capture(root, FLOOR, args.jar, args.minecraft_dir,
            shape_groups=(("holdout", registry), ("holdout", galleries)), viewpoints=partial(poses, obliques=ANGLES))
    label_capture(root)
    if args.world_out:
        archive_world(root, args.world_out.resolve())


if __name__ == "__main__":
    main()
