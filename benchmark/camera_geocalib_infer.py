"""Infer pinhole intrinsics and camera-frame up from one RGB, without pose or FoV priors."""
import argparse
import subprocess
from pathlib import Path
import numpy as np

from benchmark.expanded_baseline import ROOT, COHORTS, SOURCE, digest, frames, save

OUT = ROOT / "runs/camera-geocalib-01"
WEIGHT_SHA = "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac"
REVISION = "1dedb29f94fc11b896c5f572005978aeebccd67f"


def proposal(model, rgb):
    result = model.calibrate(model.load_image(rgb).to(next(model.parameters()).device))
    camera, gravity = result["camera"], result["gravity"]
    def values(tensor):
        return tensor.detach().cpu().flatten().tolist()
    fx, fy = values(camera.f)
    cx, cy = values(camera.c)
    gx, gy, gz = values(gravity.vec3d)
    # g dot K^-1 [x,y,1] = 0 on the projected horizon.
    return dict(focal_xy_px=[fx, fy], principal_xy_px=[cx, cy], up_cam=[gx, gy, gz],
                roll_deg=values(gravity.roll * 180 / np.pi)[0],
                pitch_up_deg=values(gravity.pitch * 180 / np.pi)[0],
                focal_uncertainty_px=values(result["focal_uncertainty"])[0],
                horizon_center_px=cy - fy * gz / gy if abs(gy) > .001 else None)


def main():
    import torch
    import geocalib
    from geocalib import GeoCalib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/geocalib/pinhole.tar")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    upstream = Path(geocalib.__file__).resolve().parents[1]
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"Unexpected GeoCalib revision {actual}")
    if digest(args.weights) != WEIGHT_SHA:
        raise ValueError("Unexpected model weight digest")
    model = GeoCalib(str(args.weights)).to(args.device)
    for entry in sorted(frames(SOURCE, COHORTS), key=lambda e: e["split"] != "train"):
        rgb = Path(entry["frame"]) / "rgb.png"
        if digest(rgb) != entry["rgb_sha256"]:
            raise ValueError(rgb)
        parameters = proposal(model, rgb)
        save(OUT / "proposals" / f"{entry['id']}.json", dict(
             **parameters, rgb_sha256=entry["rgb_sha256"], weights_sha256=WEIGHT_SHA,
             upstream_commit=REVISION, torch_version=torch.__version__))
        print(entry["id"], flush=True)


if __name__ == "__main__":
    main()
