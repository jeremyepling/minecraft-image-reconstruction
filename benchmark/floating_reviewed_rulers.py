"""Measure the six reviewed masks and rulers in native visible reference pixels."""
from collections import Counter
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from benchmark.floating_extra_feedback import CASES, paths, read, project, _native_alpha

OUT = Path(__file__).parent / "reviews/floating-wall-feedback-01"


def ruler(chart, faces):
    if not faces:
        return {}
    inverse = np.linalg.inv(np.array(chart["H"]))
    corners = np.array([project(inverse, np.array(f["corners_px"])) for f in faces])
    # Measure full projected face area in proposed cells, including a perspective gradient.
    x, y = corners.transpose(2, 0, 1)
    areas = .5 * np.abs((x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1))
    def error(region):
        xy = corners[region]
        delta = np.abs(xy - np.round(xy))
        return round(float(np.median(delta)), 3)
    left = np.asarray([np.mean(f["corners_px"], axis=0)[0] for f in faces])
    return dict(faces=len(faces), median_cells_per_face=round(float(np.median(areas)), 2),
                left_quarter_corner_error_cells=error(left <= np.quantile(left,.25)),
                right_quarter_corner_error_cells=error(left >= np.quantile(left,.75)))


def main():
    notes = read(OUT / "feedback.json")
    sheet = Image.new("RGB",(1080, 205*len(CASES)+45),"#19232f")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arial.ttf", 17)
    draw.text((10,9), "Source crop                  |    Mask composition: green principal surface, orange other surfaces, pink background", font=font)
    report = {}
    for i,(panel,name,wall,title,_) in enumerate(CASES):
        (src,seeds,expanded,cases),_,new = paths(panel)
        case = next(c for c in cases if c["id"] == name)
        frame = Path(case["frame"])
        labels = read(frame/"labels.json")
        with np.load(frame/"labels.npz") as ids:
            owner = ids["face_id"]
        faces = {f["id"]:f for f in labels["faces"]}
        blocks = {b["id"]:b for b in labels["blocks"]}
        with Image.open(case["variants"]["sky"]["image"]) as im:
            source = im.convert("RGB")
        plane = next(p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
        alpha,(x,y,r,b) = _native_alpha(plane["support"],source.size)
        mask = np.asarray(alpha)>0
        sub = owner[y:b,x:r]
        counts = dict(zip(*np.unique(sub[mask],return_counts=True)))
        surfaces, materials = Counter(), Counter()
        for fid,count in counts.items():
            if fid:
                f = faces[fid]
                surfaces[f["surface_id"]] += int(count)
                materials[blocks[f["block"]]["block_id"]] += int(count)
        principal,principal_pixels = surfaces.most_common(1)[0]
        eligible = [f for fid,f in faces.items() if f["surface_id"]==principal and
                    counts.get(fid,0) >= .5*f["visible_pixels"] and
                    np.allclose(np.linalg.norm(np.roll(f["corners_world"],-1,axis=0)-np.array(f["corners_world"]),axis=1), 1.) and all(f["corner_visible"])]
        def selected(folder):
            return next(c for c in read(folder/"sky"/name/"charts.json")["charts"] if c["id"]==wall)
        a,bgrid = [ruler(selected(folder),eligible) for folder in
                   (expanded,seeds if panel=="holdout5" else new.parent/"charts")]
        total = int(mask.sum())
        report[name] = dict(wall=wall, preference=notes[name]["preferred"], mask_pixels=total,
                            background_pixels=int(counts.get(0,0)),principal_surface_id=principal,
                            principal_pixels=principal_pixels, surfaces=surfaces, materials=materials,
                            original=a,alternative=bgrid)
        source_crop = source.crop((x,y,r,b))
        rgb = np.array(source_crop)
        painted = (rgb*.42).astype("uint8")
        principal_ids = [fid for fid,f in faces.items() if f["surface_id"]==principal]
        dom = np.isin(sub,principal_ids)
        for region,color in ((mask & dom,[46,212,136]),(mask & ~dom & (sub!=0),[255,158,46]),(mask & (sub==0),[255,75,177])):
            painted[region] = rgb[region]*.5+np.array(color)*.5
        top=42+i*205
        for j,pic in enumerate((source_crop, Image.fromarray(painted))):
            pic.thumbnail((320,190),Image.Resampling.LANCZOS)
            sheet.paste(pic,(8+328*j,top))
        desc=[title+"  /  "+wall, f"Principal surface  {100*principal_pixels/total:.1f}% of mask",
              f"Background  {100*counts.get(0,0)/total:.1f}%"]
        if a:
            desc.append(f"A / B cells per full face  {a['median_cells_per_face']} / {bgrid['median_cells_per_face']}")
        for j,line in enumerate(desc):
            draw.text((668,top+14+j*29),line,font=font,fill="white")
    sheet.save(OUT/"mask-and-scale.jpg",quality=93)
    (OUT/"measurements.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
