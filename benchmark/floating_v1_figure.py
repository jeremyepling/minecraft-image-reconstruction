"""Render source and selected-component overlays for two new and one regressing V1 view."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_v1_report import REVIEW, read, v1


def render():
    rows={r["id"]:r for r in read(REVIEW/"conditions.json.gz")}
    cases={c["id"]:c for c in v1.tasks()}
    ids=["train-arcade3-cool-high-right","legacy-smooth_stone-holdout-portico-plaster-high-right",
         "legacy-smooth_stone-dev-wing-plaster-high-right"]
    labels=["Arcade 3, cool high-right","Portico, plaster high-right","Wing, plaster high-right"]
    canvas=Image.new("RGB",(1080,690),"#18252d")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("arial.ttf",17)
    for ix,(n,title) in enumerate(zip(ids,labels)):
        r=rows[n]; y=ix*230
        draw.text((12,y+3),title,fill="white",font=font)
        src=Path(cases[n]["variants"]["sky"]["image"])
        views=[("Original source",src)]
        for arm,label in (("split-border","Baseline — expanded split-border"),("current","Guarded pre-grid")):
            proof=r["arms"][arm]
            count=proof["score"]["geometry"]["strict"]
            views.append((f"{label}  {count['correct']} correct",Path(proof["pack"]).with_name("source-overlay.jpg")))
        for j,(label,path) in enumerate(views):
            image=Image.open(path).convert("RGB")
            image.thumbnail((350,175),Image.Resampling.LANCZOS)
            canvas.paste(image,(j*360+(360-image.width)//2,y+28))
            draw.text((j*360+5,y+203),label,font=font,fill="#ddf0ed")
    canvas.save(REVIEW/"transfer-examples.jpg",quality=91)


if __name__=="__main__":
    render()
