"""Stage private worlds that replay the original Visible Builds v1 cameras at two times."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from calibration.world import GROUND, PACK, create_world


# Near the ends of daytime, vanilla changes structure lighting as well as the sky.
LIGHTING = (dict(name="morning", ticks=400), dict(name="afternoon", ticks=11600))
COHORTS = {
    "pilot": 96,
    "camera-round-01": 20,
    "camera-gray-concrete-v1": 20,
    "surface-confirm-v1": 8,
    "normal-confirm-v1": 8,
}
STATE = re.compile(r"Block\{(minecraft:[a-z0-9_]+)\}(\[[a-z0-9_=,-]+\])?")


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def original_frames(source, manifest):
    frames = {}
    for split in manifest["shards"]:
        split_dir = source / "data" / split
        for line in (split_dir / "frames.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            if row["id"] in frames or row["split"] != split or row["dataset"] != manifest["dataset"]:
                raise ValueError(f"Duplicate or unexpected original frame {row['id']}")
            folder = split_dir / "frames" / row["id"]
            if read(folder / "frame.json") != row:
                raise ValueError(f"Frame index mismatch: {folder}")
            frames[row["id"]] = (row, folder)
    expected = {row["id"]: row for row in manifest["frames"]}
    if frames.keys() != expected.keys() or any(
        ref["rgb_sha256"] != frames[frame_id][0]["files"]["rgb.png"]
        or any(ref[key] != frames[frame_id][0][key] for key in ("layout", "split", "palette"))
        for frame_id, ref in expected.items()
    ):
        raise ValueError("Extracted frame index differs from the published Visible Builds v1 index")
    return tuple(frames.values())


def checked_sources(row, folder):
    for name in ("camera.json", "world-readback.json"):
        if digest(folder / name) != row["files"][name]:
            raise ValueError(f"Original frame checksum mismatch: {folder / name}")
    pose, readback = read(folder / "camera.json"), read(folder / "world-readback.json")
    if (pose["minecraft_version"] != "26.3" or pose["clock_ticks"] != 6000
        or pose["image_size"] != [1440, 900] or pose["vertical_fov_deg"] != 55
        or pose["resource_packs"] != ["vanilla"] or pose["raining"] or pose["thundering"]):
        raise ValueError(f"Unexpected source camera settings: {folder}")
    states = {}
    for block in readback:
        point = tuple(block[axis] for axis in "xyz")
        match = STATE.fullmatch(block["state"])
        if (point in states or not match or match[1] != "minecraft:" + block["block_id"]
            or point[1] < GROUND or (point[1] == GROUND and block["block_id"] != row["floor"])):
            raise ValueError(f"Unexpected source state at {point}: {folder}")
        states[point] = match[1] + (match[2] or "")
    return pose, states


def stage(source, jar, output, cohorts):
    manifest = read(Path(__file__).with_name("visible-builds-v1.json"))
    source_frames = original_frames(source, manifest)
    records = source / manifest["capture_records"]["file"]
    if digest(records) != manifest["capture_records"]["sha256"]:
        raise ValueError(f"Original capture record checksum mismatch: {records}")
    with ZipFile(records) as archive:
        options = json.loads(archive.read("saved-client-options.json"))
    with ZipFile(jar) as archive:
        metadata = json.loads(archive.read("version.json"))
    client_sha256 = digest(jar)
    if metadata["id"] != manifest["minecraft_version"] or any(
        row["client_sha256"] != client_sha256 or row["ground_y"] != GROUND for row, _ in source_frames
    ):
        raise ValueError("Replay requires the original Minecraft 26.3 client and staging ground")
    for cohort in cohorts:
        frames = [(row, folder) for row, folder in source_frames if row.get("legacy_fixture", "pilot") == cohort]
        if len(frames) != COHORTS[cohort]:
            raise ValueError(f"Unexpected number of original frames in {cohort}")
        stage_world(output / cohort, frames, metadata, client_sha256, options)


def stage_world(root, frames, metadata, client_sha256, options):
    floors = {row["floor"] for row, _ in frames}
    if len(floors) != 1:
        raise ValueError("Every replay world must use one staging floor")
    floor, = floors
    root.mkdir(parents=True)
    world = root / "fixture/saves/Palette Lab"
    create_world(world, metadata, {"minecraft:random_tick_speed": 0}, level_name="Visible Builds v1 replay",
                 floor=floor)
    (world / "datapacks").mkdir()

    sites = defaultdict(list)
    for row, folder in frames:
        sites[row["layout"], row["palette"]].append((row, folder))
    bounds = set()
    blocks, views = [], []
    with ZipFile(world / "datapacks" / PACK, "w", ZIP_DEFLATED) as pack:
        fmt = [metadata["pack_version"][key] for key in ("data_major", "data_minor")]
        pack.writestr("pack.mcmeta", json.dumps(dict(pack=dict(description="Visible Builds v1 lighting replay",
                                                              min_format=fmt, max_format=fmt))))
        for page, site in enumerate(sorted(sites.values(), key=lambda site: site[0][0]["origin_world"])):
            states = None
            for row, folder in site:
                pose, readback = checked_sources(row, folder)
                if states is not None and readback != states:
                    raise ValueError(f"Source block states change across cameras in {row['id']}")
                states = readback
                x, y, z = pose["eye_world"]
                views.append(dict(id=row["id"], page=page, x=x, feetY=y - 1.62, z=z,
                                  yawPitch=pose["yaw_pitch_deg"], lighting=LIGHTING))
            overlap = bounds & states.keys()
            if overlap:
                raise ValueError(f"Original pages overlap at {next(iter(overlap))} in {root.name}")
            bounds.update(states)
            commands = []
            for (x, y, z), state in sorted(states.items()):
                block_id = state.split("[", 1)[0].removeprefix("minecraft:")
                blocks.append(dict(page=page, x=x, y=y, z=z, block_id=block_id, expected_state=state))
                if y > GROUND and state != "minecraft:air":
                    commands.append(f"setblock {x} {y} {z} {state} strict")
            pack.writestr(f"data/palette_lab/function/page_{page}.mcfunction", "\n".join(commands) + "\n")
    write(root / "views.json", views)
    write(root / "layout.json", dict(version=metadata["id"], client_sha256=client_sha256,
                                     ground_y=GROUND, floor=floor, blocks=blocks))
    write(root / "capture-options.json", options)
    print(f"Staged {len(views)} exact source cameras at two times in {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted original Visible Builds v1 release")
    parser.add_argument("--jar", type=Path, required=True, help="Original Minecraft 26.3 client JAR")
    parser.add_argument("--out", type=Path, required=True, help="Root for the private cohort capture directories")
    parser.add_argument("--cohort", choices=COHORTS, action="append", help="Only stage these cohorts")
    args = parser.parse_args()
    stage(args.source, args.jar, args.out, args.cohort or COHORTS)


if __name__ == "__main__":
    main()
