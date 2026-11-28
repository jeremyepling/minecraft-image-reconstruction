"""Render the reviewed source partitions and proposed native grid boundaries."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_pregrid_depth_refit import OUT, REVIEW, project, read, workload


def main():
    rows = [
        ("train-stable-yard-1-warm-high-right", "mask-0", 24, "complete", "Stable Yard 1 high-right", "21 verified eave cells. Component stays detached."),
        ("train-stable-yard-1-warm-low-left", "mask-0", None, None, "Stable Yard 1 low-left", "No verified local grid. Guard keeps the original build."),
        ("train-pillars-01-marble-low-left", "mask-1", 7, "complete", "Pillars low-left", "31 proposed cells. Winning component excludes this wall."),
        ("train-arcade1-cool-high-right", "mask-1", 1, "partial", "Arcade 1 high-right", "14 proposed cells. Shared edges close a three-surface corner."),
    ]
    jobs = {j[1]["id"]:j for j in workload()}
    canvas = Image.new("RGB",(1000, len(rows)*228), "#152028")
    caption = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf",17)
    for index,(case_id,parent,part,arm,name,note) in enumerate(rows):
        panel,case,*_ = jobs[case_id]
        source = Image.open(case["variants"]["sky"]["image"]).convert("RGB")
        labels = np.array(Image.open(OUT/panel/"source"/case_id/(parent+"-ids.png")))
        rgb = np.array(source)
        y,x = np.nonzero(labels)
        bbox = (max(0,x.min()-8), max(0,y.min()-8), min(rgb.shape[1],x.max()+9), min(rgb.shape[0],y.max()+9))
        colors=np.random.default_rng(17).integers(40,255,(int(labels.max())+1,3))
        take=labels>0
        rgb[take]=.52*rgb[take]+.48*colors[labels[take]]
        marked=Image.fromarray(rgb)
        if part is not None:
            packet=read(OUT/panel/arm/"charts/sky"/case_id/"charts.json")
            chart=next(c for c in packet["charts"] if c["id"]==f"{parent}~p{part}")
            draw=ImageDraw.Draw(marked)
            for cell in chart["cells"]:
                u,v=cell["u"],cell["v"]
                corners=np.array([[u,v],[u+1,v],[u+1,v+1],[u,v+1],[u,v]])
                points=project(np.array(chart["H"]),corners)
                draw.line([tuple(p) for p in points],fill="#61ff55",width=2)
        top=index*228
        caption.text((10,top+4),name+"   |   "+note,fill="white",font=font)
        for col,img in enumerate((source,marked)):
            crop=img.crop(bbox)
            crop.thumbnail((485,188),Image.Resampling.LANCZOS)
            canvas.paste(crop,(col*500+(500-crop.width)//2,top+32))
    canvas.save(REVIEW/"source-partitions.jpg",quality=92)


if __name__=="__main__":
    main()
