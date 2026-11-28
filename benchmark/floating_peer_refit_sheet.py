"""Render projected candidate grids on the same source crops as the reference-free baseline."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_peer_ruler_refit import BASE, OUT, REVIEW, PREFIX, project, read, work
from engine.geometry import UNIT


def main():
    rows=[r for r in read(REVIEW/"conditions.json.gz") if r["audit"]["charts"]]
    rows.sort(key=lambda r:("stable" not in r["id"],"low-left" not in r["id"]))
    manifest={(g,c["id"]):c for g,c,_ in work()}
    atlas_rows={r["id"]:r for r in read(REVIEW/"atlas-first/conditions.json.gz")}
    image=Image.new("RGB",(1260,235*len(rows)),"white")
    draw=ImageDraw.Draw(image)
    title=ImageFont.truetype("arial.ttf",17)
    label=ImageFont.truetype("arial.ttf",14)
    for i,r in enumerate(rows):
        group,name=r["audit"]["group"],r["id"]
        src=Image.open(manifest[group,name]["variants"]["sky"]["image"]).convert("RGB")
        cid=next(iter(r["audit"]["charts"]))
        src_charts=read(BASE/group/"large-contraction/charts/sky"/name/"charts.json")["charts"]
        chart=next(c for c in src_charts if c["id"]==cid)
        xy=project(np.array(chart["H"]),np.array([(c["u"]+.5,c["v"]+.5) for c in chart["cells"]]))
        box=tuple(map(int,(*np.maximum(0,xy.min(0)-20),*np.minimum(src.size,xy.max(0)+20))))
        imgs=[src]
        for directory in (BASE/group/"large-contraction",OUT/group/"relaxed",OUT/"atlas-first"/group/"relaxed"):
            imgs.append(Image.open(directory/"assembly"/PREFIX/name/"source-overlay.jpg").convert("RGB"))
        a=atlas_rows[name]
        chosen=next(iter(a["audit"]["charts"].values()))
        corners=project(np.array(chosen["homography"]),UNIT)
        ImageDraw.Draw(imgs[-1]).line([tuple(p) for p in [*corners,corners[0]]],fill="#f8941c",width=4)
        draw.text((5,235*i+2),name.replace("train-",""),font=title,fill="black")
        for j,(im,text) in enumerate(zip(imgs,("Source","Whole wall","Edge refit","Atlas-first refit"))):
            if j:
                score=(a if j==3 else r)["score"]["base" if j==1 else "relaxed"]["geometry"]["strict"]
                text+="    "+"/".join(str(score[k]) for k in ("correct","missed","extras"))
            im=im.crop(box)
            im.thumbnail((310,174),Image.Resampling.LANCZOS)
            image.paste(im,(315*j+(315-im.width)//2,235*i+30))
            draw.text((315*j+5,235*i+208),text,fill="black",font=label)
    image.save(REVIEW/"replacement-rulers.jpg",quality=92)


if __name__=="__main__":
    main()
