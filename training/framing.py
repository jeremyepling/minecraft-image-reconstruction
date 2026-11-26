"""Admit count-training images only when the complete source build fits with a border."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np

from benchmark.camera_labels import screen_matrix


MIN_MARGIN_FRACTION = 0.02
SIDES = ("left", "top", "right", "bottom")
CUBE_CORNERS = np.asarray(list(product((0, 1), repeat=3)))


def measure_framing(pose, readback, ground_y, face_ids):
    """Use all occupied source cubes for framing only. Visible labels are still the targets."""
    width, height = pose["image_size"]
    if face_ids.shape != (height, width):
        raise ValueError("Face map must have the native camera resolution")
    required = np.array([width, height, width, height]) * MIN_MARGIN_FRACTION
    occupied = [(b["x"], b["y"], b["z"]) for b in readback
                if b["y"] > ground_y and b["block_id"].removeprefix("minecraft:") != "air"]
    result = dict(image_size=[width, height], occupied_cells=len(occupied), accepted=False,
                  projected_margins_px=None, visible_margins_px=None)
    mask = face_ids > 0
    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    if len(columns):
        # Count whole empty columns/rows between the labeled silhouette and the image boundary.
        visible = np.array([columns[0], rows[0], width-1-columns[-1], height-1-rows[-1]])
        result["visible_margins_px"] = visible.tolist()
    if not occupied:
        return dict(result, reason="empty_build")

    # With positive depths the extremal projected coordinate of a cube is at one of
    # its eight vertices. Occupied cells may be disjoint or completely off screen.
    corners = (np.asarray(occupied)[:, None, :] + CUBE_CORNERS).reshape(-1, 3)
    projection = screen_matrix(pose)
    screen = (corners - pose["eye_world"]) @ projection[:, :3].T + projection[:, 3]
    if not np.isfinite(screen).all():
        raise ValueError("Non-finite projected source geometry")
    if np.any(screen[:, 2] <= 0):
        return dict(result, reason="at_or_behind_camera")
    xy = screen[:, :2] / screen[:, 2:3]
    left, top = xy.min(axis=0)
    right, bottom = (width, height) - xy.max(axis=0)
    projected = np.array([left, top, right, bottom])
    result["projected_margins_px"] = projected.round(4).tolist()
    if not len(columns):
        return dict(result, reason="no_visible_build_pixels")
    if np.any(projected < 0):
        return dict(result, reason="clipped")
    if np.any(projected < required) or np.any(visible < required):
        return dict(result, reason="insufficient_border")
    return dict(result, accepted=True, reason="whole_structure")


def check_frame(frame_dir):
    """Apply the v2 rule to one unpacked frame directory."""
    def read(name):
        return json.loads((frame_dir / name).read_bytes())

    with np.load(frame_dir / "labels.npz", allow_pickle=False) as raster:
        return measure_framing(read("camera.json"), read("world-readback.json"),
                               read("frame.json")["ground_y"], raster["face_id"])


def audit(data):
    frames, counts = [], {}
    for split in ("train", "validation", "test", "regression"):
        reasons = Counter()
        for line in (data / split / "frames.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            frame_dir = data / split / row["directory"]
            if row != json.loads((frame_dir / "frame.json").read_bytes()) or row["split"] != split:
                raise ValueError(f"Frame index disagrees with {frame_dir}")
            result = check_frame(frame_dir)
            reasons[result["reason"]] += 1
            frames.append(dict(id=row["id"], split=split, rgb_sha256=row["files"]["rgb.png"], **result))
        counts[split] = dict(total=reasons.total(), kept=reasons["whole_structure"],
                             removed=reasons.total()-reasons["whole_structure"], reasons=dict(reasons))
    return dict(min_margin_fraction=MIN_MARGIN_FRACTION, margin_order=SIDES,
                splits=counts, frames=frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Root containing the unpacked splits")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))
