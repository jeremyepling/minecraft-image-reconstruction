"""Check native Floating Front captures and package first-hit visible labels for each frame."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from benchmark.camera_fixture import VERSION, write
from engine.atlas import digest
from training.floating_front_release import DATASET
from training.decorative_labels import properties_for, write_labels as write_partial
from training.framing import measure_framing
from training.visible_labels import write_labels as write_full

FILES = ("rgb.png", "camera.json", "world-readback.json", "labels.json", "labels.npz")
CAMERA_FIELDS = ("eye_world", "yaw_pitch_deg", "vertical_fov_deg", "view_rotation_projection", "image_size")


def read(path):
    return json.loads(path.read_bytes())


def check_scene(view, pose, rows, expected):
    if (pose["minecraft_version"] != VERSION or pose["id"] != view["id"] or pose["raining"]
            or pose["thundering"] or pose["clock_ticks"] % 24000 != view["clock_ticks"]
            or pose["clouds"] not in ("false", "off")
            or pose["resource_packs"] != ["vanilla"] or not pose["hud_hidden"]):
        raise ValueError(f"Unexpected native conditions for {view['id']}: {pose}")
    if "client_skylight_checked_cells" in pose and pose["client_skylight_checked_cells"] != len(rows):
        raise ValueError(f"Incomplete native client skylight readback in {view['id']}")
    states = {(r["x"], r["y"], r["z"]): r for r in rows}
    if len(states) != len(rows) or states.keys() != expected.keys():
        raise ValueError(f"Missing or duplicated server readbacks in {view['id']}")
    for point, entry in expected.items():
        actual = states[point]
        if actual["block_id"] != entry["block_id"]:
            raise ValueError(f"Incorrect block {view['id']} {point}: {actual} != {entry}")
        _, bracket, properties = entry["expected"].partition("[")
        if bracket and properties_for(actual) != dict(p.split("=") for p in properties[:-1].split(",")):
            raise ValueError(f"Incorrect properties {view['id']} {point}: {actual} != {entry}")
    return {p: row for p, row in states.items() if row["block_id"] != "air"}


def finalize(root, jar):
    layout = read(root / "layout.json")
    if digest(jar) != layout["client_sha256"]:
        raise ValueError("Label assets and capture jar differ")
    pages = {}
    for row in layout.pop("blocks"):
        pages.setdefault(row["page"], {})[(row["x"], row["y"], row["z"])] = row
    frames = root / "frames"
    frames.mkdir(exist_ok=True)
    for view in read(root / "views.json"):
        fid = view["id"]
        capture = root / "captures"
        pose = read(capture / f"{fid}.json")
        rows = read(capture / f"{fid}-blocks.json")
        states = check_scene(view, pose, rows, pages[view["page"]])
        parent = frames / view["parent_id"]
        if fid != view["parent_id"]:
            parent_pose = read(parent / "camera.json")
            if any(parent_pose[k] != pose[k] for k in CAMERA_FIELDS):
                raise ValueError(f"Camera changed across pair: {fid}")
        if (frames / fid / "frame.json").exists():
            row = read(frames / fid / "frame.json")
            if all(digest(frames / fid / file) == row["files"][file] for file in FILES):
                continue
            raise ValueError(f"Existing finalized frame changed: {fid}")
        with TemporaryDirectory(dir=frames) as tmp:
            frame = Path(tmp) / fid
            frame.mkdir()
            for path, name in ((f"{fid}.png", "rgb.png"), (f"{fid}.json", "camera.json"),
                               (f"{fid}-blocks.json", "world-readback.json")):
                shutil.copyfile(capture / path, frame / name)
            with Image.open(frame / "rgb.png") as native:
                native.load()
                if list(native.size) != pose["image_size"] or native.format != "PNG":
                    raise ValueError(f"Image does not match camera: {fid}")
            if view["condition"] == "decorative":
                counts = write_partial(pose, states, view["origin_world"], frame, jar=jar)
                if not {"fence", "trapdoor", "slab", "stairs"} <= counts["visible_block_kinds"].keys():
                    raise ValueError(f"Missing visible decoration: {fid}")
            elif view["lighting"] == "noon":
                counts = write_full(pose, states, view["origin_world"], frame)
            else:
                counts = read(parent / "frame.json")["counts"]
                labels = read(parent / "labels.json")
                labels["frame_id"] = fid
                write(frame / "labels.json", labels)
                shutil.copyfile(parent / "labels.npz", frame / "labels.npz")
            counts = {k: v for k, v in counts.items() if k.startswith("visible_")}
            with np.load(frame / "labels.npz") as rasters:
                framing = measure_framing(pose, rows, 64, rasters["face_id"])
            if not framing["accepted"]:
                raise ValueError(f"Native framing failed: {fid} {framing}")
            widths = np.array(pose["image_size"])
            margins = np.array(framing["visible_margins_px"])
            framing["longest_axis_fraction"] = float(max(1 - (margins[:2] + margins[2:]) / widths))
            if framing["longest_axis_fraction"] < .75:
                raise ValueError(f"Structure is too small in frame: {fid} {framing}")
            fields = ("id", "layout", "family", "size", "split", "palette", "angle", "angle_deg",
                      "elevation_deg", "lighting", "clock_ticks", "condition", "dimensions",
                      "origin_world", "parent_id", "number")
            info = {k: view[k] for k in fields}
            # Earlier standard captures have neither marker. The first reload pilot
            # recorded its verified client-cell count before the named mode existed.
            mode = pose.get("capture_lighting_mode", "reload" if "client_skylight_checked_cells" in pose else "standard")
            info.update(capture_lighting_mode=mode, dataset=DATASET, version=VERSION, client_sha256=layout["client_sha256"],
                        ground_y=64, floor="void", directory="frames/" + fid, counts=counts,
                        framing=framing, files={file: digest(frame / file) for file in FILES})
            write(frame / "frame.json", info)
            frame.rename(frames / fid)
        print(f"Verified {fid}: {counts['visible_blocks']} visible blocks", flush=True)
