"""Package or independently verify the playable surface-confirm-v1 world."""
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from benchmark.world_archives import audit, load, pack
from engine.atlas import digest
from engine.nbt import read_nbt


ROOT = Path(__file__).resolve().parent
RELEASE = "surface-confirm-v1"
OUTPUT = ROOT / "worlds" / RELEASE
FIXTURE = ROOT / "data" / RELEASE
TITLE = "Surface confirmation v1 - Workshop and crossing"
FILENAME = "workshop-and-crossing.zip"


def world_record(archive, fixture):
    floor = load(fixture / "layout.json")["floor"]
    record = audit(archive, TITLE, fixture, floor)
    with ZipFile(archive) as zf:
        private = {"players", "playerdata", "advancements", "stats", "session.lock", "level.dat_old"}
        for entry in zf.namelist():
            path = PurePosixPath(entry)
            if path.parts[0] != TITLE or ".." in path.parts or private.intersection(path.parts):
                raise ValueError(f"Unexpected entry in portable save: {entry}")
        data = read_nbt(gzip.decompress(zf.read(f"{TITLE}/level.dat")))["Data"]
        if "Player" in data or "singleplayer_uuid" in data:
            raise ValueError("World metadata contains a player identity")
    return record


def index(archive, fixture):
    views = load(fixture / "views.json")
    sites = []
    for building in load(fixture / "buildings.json"):
        view = next(v for v in views if v["page"] == building["page"] and v["angle"] == "low-left")
        pose = load(fixture / "captures" / (view["id"] + ".json"))
        yaw, pitch = pose["yaw_pitch_deg"]
        sites.append(dict(layout=building["layout"], palette=building["palette"], split=building["split"],
                          bounds_xyz=building["bounds"], low_left_view=view["id"],
                          teleport=f"/tp @s {view['x']:.5f} {view['feetY']:.5f} {view['z']:.5f} {yaw:.5f} {pitch:.5f}"))
    return dict(schema=1, fixture=RELEASE, fixture_index_sha256=digest(fixture / "index.json"),
                game="Minecraft Java Edition", version="26.3",
                install=f"Extract the ZIP into your Java game directory's saves folder. Open '{TITLE}' in Singleplayer.",
                bounds="World XYZ, lower inclusive and upper exclusive.",
                worlds=[world_record(archive, fixture)], sites=sites)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, help="Package a completed capture with its saved client world")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="Archive directory (fresh directory when packaging)")
    args = parser.parse_args()
    if args.capture:
        if args.out.exists():
            raise FileExistsError(f"Use a fresh directory for a new archive: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as temp:
            release = Path(temp) / RELEASE
            release.mkdir()
            archive = release / FILENAME
            pack(args.capture / "client/saves/Palette Lab", archive, TITLE)
            verification = index(archive, args.capture)
            (release / "world-index.json").write_text(json.dumps(verification, indent=2) + "\n",
                                                      encoding="utf8", newline="\n")
            release.rename(args.out)
    else:
        verification = index(args.out / FILENAME, FIXTURE)
        if load(args.out / "world-index.json") != verification:
            raise ValueError("Portable world and frozen fixture differ from the archive index")
    print(json.dumps(verification["worlds"][0], indent=2))


if __name__ == "__main__":
    main()
