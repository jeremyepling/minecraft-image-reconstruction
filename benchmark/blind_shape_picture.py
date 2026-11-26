"""Draw a native photo, per-pixel position correctness and projected extra centers."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import read


def font(size, bold=False):
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(family, size)
    except OSError:
        return ImageFont.load_default(size=size)


def render(root, predictions, key, out):
    labels = read(root / "truth" / key / "labels.json")
    masks = np.load(root / "truth" / key / "labels.npz")["face_id"]
    photo = Image.open(root / "scoring" / "captures" / (key + ".png")).convert("RGB")
    truth = {tuple(block["xyz_world"]) for block in labels["blocks"]}
    placed = set(map(tuple, read(predictions / key / "cells.json")["world"]))
    shift, hits = best_translation(placed, truth)
    placed = {tuple(a+b for a,b in zip(cell,shift)) for cell in placed}
    misses = truth - placed
    extras = sorted(placed - truth)
    rgb = np.asarray(photo).copy()
    color = np.zeros((len(labels["faces"]) + 1, 3), np.uint8)
    for face in labels["faces"]:
        color[face["id"]] = (255,162,32) if tuple(face["xyz_world"]) in misses else (15,179,172)
    surface = masks > 0
    rgb[surface] = .33 * rgb[surface] + .67 * color[masks[surface]]
    boundaries = surface & ((masks != np.roll(masks,1,0)) | (masks != np.roll(masks,1,1)))
    rgb[boundaries] = (46,51,55)
    marked = photo.copy()
    draw = ImageDraw.Draw(marked)
    centers = np.asarray(extras, dtype=float).reshape((-1,3)) + .5
    camera = np.asarray(labels["camera"]["world_to_pixel_homogeneous"])
    proj = centers @ camera[:,:3].T + camera[:,3]
    assert np.all(proj[:,2] > 0)
    pts = proj[:,:2] / proj[:,2,None]
    for x,y in pts:
        draw.ellipse((x-8,y-8,x+8,y+8), fill=(159,18,104), outline="white", width=2)
    ys,xs = np.nonzero(surface)
    x0 = min(xs.min(), pts[:,0].min()) if extras else xs.min()
    x1 = max(xs.max(), pts[:,0].max()) if extras else xs.max()
    y0 = min(ys.min(), pts[:,1].min()) if extras else ys.min()
    y1 = max(ys.max(), pts[:,1].max()) if extras else ys.max()
    bounds = (int(x0)-28,int(y0)-28,int(x1)+29,int(y1)+29)
    height = 460
    width = round(height*(bounds[2]-bounds[0])/(bounds[3]-bounds[1]))
    pad = 23
    total = pad*4+width*3
    sheet = Image.new("RGB",(total, height+173),(248,249,250))
    draw = ImageDraw.Draw(sheet)
    text = (31,37,43)
    draw.text((pad,13),"Switchback landing   /   High-right view",font=font(30,True),fill=text)
    f1 = 2*hits/(len(placed)+len(truth))
    draw.text((pad,53),f"{len(truth)} visible target blocks   ·   {len(placed)} placed   ·   {f1:.2%} position F1 after whole-build alignment",
              font=font(20),fill=text)
    titles = ("Source image", "Visible target faces", "Extra placements")
    for i,(image,title) in enumerate(zip((photo,Image.fromarray(rgb),marked),titles)):
        x = pad + (pad+width)*i
        draw.text((x,96),title,font=font(22,True),fill=text)
        sheet.paste(image.crop(bounds).resize((width,height),Image.Resampling.LANCZOS),(x,129))
    y = height + 141
    draw.text((pad,y),"Same crop in all three panels",font=font(17),fill=text)
    x = pad*2 + width
    for swatch,label in (((15,179,172),f"{hits} correct"),((255,162,32),f"{len(misses)} missed")):
        draw.rectangle((x,y+2,x+15,y+17),fill=swatch)
        draw.text((x+23,y-1),label,font=font(17),fill=text)
        x += 150
    x = pad*3 + width*2
    draw.ellipse((x,y+2,x+15,y+17),fill=(159,18,104))
    draw.text((x+23,y-1),f"{len(extras)} centers projected; dots can overlap",font=font(17),fill=text)
    sheet.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.predictions, args.id, args.out)
