"""Grade cached RGB predictions, decoded exports and the matched-budget fallback."""
from pathlib import Path
from collections import defaultdict
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.camera_geocalib_run import OUT, ROOT, CONTROL, BASE, REPEAT
from benchmark.camera_geocalib_infer import WEIGHT_SHA, REVISION
from benchmark.camera_horizon_report import ground_validation
from benchmark.camera_metrics import truth_camera
from benchmark.expanded_baseline import frames, SOURCE, COHORTS, read, save, digest
from benchmark.expanded_scores import summarize
from engine.geometry import project

REVIEW = ROOT/"benchmark/reviews/camera-geocalib-01"


def error_metrics(entry, proposal):
    pose = read(Path(entry["frame"])/"camera.json")
    w, h = pose["image_size"]
    Q = truth_camera(pose)
    f = h / (2*np.tan(np.radians(pose["vertical_fov_deg"])/2))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    up = np.linalg.solve(K, Q[:,1])
    up /= np.linalg.norm(up)
    true_horizon = h/2-f*up[2]/up[1]
    predicted_f = proposal["focal_xy_px"][1]
    return dict(fov_abs_deg=abs(float(np.degrees(2*np.arctan(h/(2*predicted_f))))-pose["vertical_fov_deg"]),
                up_angle_deg=float(np.degrees(np.arccos(np.clip(up @ proposal["up_cam"], -1, 1)))),
                horizon_abs_px=abs(proposal["horizon_center_px"]-true_horizon),
                true_horizon_px=float(true_horizon))


def draw_example(entry, row, learned):
    from benchmark.camera_metrics import registration
    folder = Path(entry["frame"])
    image = Image.open(folder/"rgb.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    h, w = image.height, image.width
    pred = row["proposal"]
    hz = pred["horizon_center_px"]
    gx, gy, _ = pred["up_cam"]
    fx, fy = pred["focal_xy_px"]
    slope = -gx/gy*fy/fx
    draw.line((0,hz-slope*w/2,w,hz+slope*w/2), fill="orange", width=4)
    actual = row["proposal_error"]["true_horizon_px"]
    draw.line((0,actual,w,actual), fill="cyan", width=3)
    if learned["P"] is not None:
        P = np.asarray(learned["P"])
        aligned = P@np.linalg.inv(registration(P, read(folder/"camera.json")))
        corners = {tuple(c):xy for face in read(folder/"labels.json")["faces"]
                   for c,xy,seen in zip(face["corners_world"],face["corners_px"],face["corner_visible"]) if seen}
        for xyz,xy in list(corners.items())[::max(1,len(corners)//80)]:
            x,y = xy
            u,v = project(aligned,[xyz])[0]
            draw.ellipse((x-3,y-3,x+3,y+3),outline="cyan",width=2)
            draw.line((u-3,v,u+3,v),fill="magenta",width=2)
            draw.line((u,v-3,u,v+3),fill="magenta",width=2)
    draw.rectangle((0,0,w,38),fill="black")
    draw.text((8,5),entry["id"]+" | orange learned horizon   cyan recorded horizon/corners   magenta fitted corners",
              font=ImageFont.load_default(size=22))
    return image.resize((900,round(900*h/w)))


def main():
    base = read(CONTROL/"benchmark/reviews/expanded-baseline-01/cases.json")
    entries = {e["id"]:e for e in frames(SOURCE,COHORTS)}
    all_rows = []
    options = defaultdict(list)
    fixed = CONTROL/"benchmark/reviews/typed-structures-01/typed-exports.zip"
    with ZipFile(fixed) as packs:
        for row in base:
            key, dataset = row["id"], row["dataset"]
            pack = (sha256(packs.read(key+"/reconstruction.zip")).hexdigest() if dataset=="fixed-104"
                    else digest(BASE/dataset/key/"reconstruction.zip"))
            if pack != row["pack_sha256"]:
                raise ValueError(key)
            choices = dict.fromkeys(("ordinary","repeat","geocalib","matched"), row)
            result = {k:row[k] for k in ("id","dataset","split","source_sha256")}
            if key in entries:
                pred = read(OUT/"proposals"/(key+".json"))
                if pred["rgb_sha256"] != row["source_sha256"]:
                    raise ValueError(key)
                result.update(proposal=pred, proposal_error=error_metrics(entries[key],pred))
            if not row["camera_usable"]:
                old = read(REPEAT/key/"repeat.json")["camera"]
                if old["P"] is not None:
                    repeated = read(REPEAT/key/"repeat-score.json")
                    if repeated["pack_sha256"] != digest(REPEAT/key/"repeat/reconstruction.zip"):
                        raise ValueError(key)
                    choices.update(dict.fromkeys(("repeat","geocalib","matched"),repeated))
                else:
                    for name, folder in (("geocalib",OUT),("matched",OUT/"matched-control")):
                        camera = read(folder/"metric"/(key+".json"))["camera"]
                        result[name+"_camera"] = camera
                        if camera["P"] is not None:
                            scored = read(folder/"scores"/(key+".json"))
                            if scored["pack_sha256"] != digest(folder/"builds"/dataset/key/"reconstruction.zip"):
                                raise ValueError(key)
                            placed = read(folder/"builds"/dataset/key/"build-manifest.json")["placement"]
                            if placed["P"] != camera["P"]:
                                raise ValueError(f"Rebuilt camera differs: {key}")
                            choices[name] = scored
                            result[name+"_validation"] = scored["camera_validation"]|ground_validation(entries[key], camera)
            for name, score in choices.items():
                options[name].append(score)
                result[name] = {k:score[k] for k in ("geometry","scores","camera_usable","pack_sha256")}
            all_rows.append(result)
    save(REVIEW/"cases.json.gz",all_rows)
    summary = {name:summarize(rows,list(entries.values())) for name,rows in options.items()}
    save(REVIEW/"summary.json",dict(arms=summary,weights_sha256=WEIGHT_SHA,upstream_commit=REVISION,
                                    cases_sha256=digest(REVIEW/"cases.json.gz")))
    examples = ["validation-harbor-02-kiln-high-right",
                next(r["id"] for r in all_rows if r["split"]=="test" and "geocalib_camera" in r)]
    images = []
    for key in examples:
        row = next(r for r in all_rows if r["id"]==key)
        images.append(draw_example(entries[key],row,row["geocalib_camera"]))
    combined = Image.new("RGB",(images[0].width,sum(i.height for i in images)))
    combined.paste(images[0],(0,0))
    combined.paste(images[1],(0,images[0].height))
    combined.save(REVIEW/"source-examples.jpg",quality=90)


if __name__ == "__main__":
    main()
