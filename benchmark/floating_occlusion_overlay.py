"""Draw the reviewed Twinspire brick fragments and their native ownership."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_occlusion_fragments import (
    PREVIOUS_ROW, REVIEW, REVIEWED, paths, read, source_candidates, faces_at_depth,
)
from engine.geometry import project
from pathlib import Path


def render():
    source, _, _, cases = paths("original")[0]
    case = next(c for c in cases if c["id"] == REVIEWED)
    rgb = np.asarray(Image.open(case["variants"]["sky"]["image"]).convert("RGB"))
    chart = next(c for c in read(PREVIOUS_ROW/"charts/sky"/REVIEWED/"charts.json")["charts"] if c["id"]=="mask-2")
    plane = next(p for p in read(source/"sky"/REVIEWED/"source-handoff/surfaces.json")["planes"] if p["id"]=="mask-2")
    cells, witnesses = source_candidates(chart, plane, rgb)
    frame = Path(case["frame"])
    with np.load(frame/"labels.npz") as data:
        owners = data["face_id"]
    labels = read(frame/"labels.json")
    materials = {b["id"]:b["block_id"] for b in labels["blocks"]}
    planes = faces_at_depth(labels)
    principal = tuple(read(REVIEW/"paired.json.gz")["results"][0]["native"][0]["plane"])
    ids = [f["id"] for f in labels["faces"] if planes.get(f["id"])==principal and materials[f["block"]]=="minecraft:bricks"]
    target = np.isin(owners,ids)
    boundary = np.zeros_like(target)
    boundary[:,1:] |= owners[:,1:]!=owners[:,:-1]
    boundary[:,:-1] |= owners[:,1:]!=owners[:,:-1]
    boundary[1:] |= owners[1:]!=owners[:-1]
    boundary[:-1] |= owners[1:]!=owners[:-1]
    target_edge = rgb.copy()
    target_edge[boundary & target] = [255,214,47]
    for yy, xx in witnesses:
        target_edge[yy,xx] = [46,232,161]
    def outline(c):
        return project(np.asarray(chart["H"]), np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])+[c["u"],c["v"]])
    raw = Image.fromarray(rgb)
    marked = raw.copy()
    for c in chart["cells"]:
        ImageDraw.Draw(marked).line(list(map(tuple,outline(c))),fill="#779998")
    for c in cells:
        ImageDraw.Draw(marked).line(list(map(tuple,outline(c))),fill="#00ffee",width=2)
    extent = np.concatenate([outline(c) for c in chart["cells"]+cells])
    box = tuple(np.r_[np.floor(extent.min(0))-9,np.ceil(extent.max(0))+9].astype(int))
    panels = [raw, marked, Image.fromarray(target_edge)]
    crops = [im.crop(box).resize(((box[2]-box[0])*2,(box[3]-box[1])*2),Image.Resampling.NEAREST) for im in panels]
    w,h = crops[0].size
    sheet = Image.new("RGB",(3*w,h+70),"#162437")
    draw=ImageDraw.Draw(sheet)
    for i, (crop, title, caption) in enumerate(zip(crops,("Source","Five new cells","Visible brick faces"),(
            "Twinspire, high right","Cyan marks additions to the existing grid",
            "Gold marks native edges. Green marks sampled brick."))):
        sheet.paste(crop,(i*w,70))
        draw.text((i*w+10,6),title,font=ImageFont.truetype("arialbd.ttf",23),fill="white")
        draw.text((i*w+10,41),caption,font=ImageFont.truetype("arial.ttf",15),fill="#c7d9e6")
    sheet.save(REVIEW/"twinspire-fragments.png",optimize=True)


if __name__=="__main__":
    render()
