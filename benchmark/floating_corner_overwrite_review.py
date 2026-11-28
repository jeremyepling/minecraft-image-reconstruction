"""Render targeted chart overwrite comparisons for browser and mobile review."""
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_grid_signoff import WORKTREES, make
from benchmark.floating_second_signoff import option

REVIEW = Path(__file__).parent/"reviews/floating-corner-overwrite-review-01"
BG = "#14202b"
TEXT = "#edf3f7"
MUTED = "#bdcfda"
GOLD = "#ffe093"


def font(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)


def draw_wrap(draw, xy, text, width, size, color=TEXT, bold=False):
    face=font(size,bold)
    line=""
    x,y=xy
    for word in text.split():
        trial=f"{line} {word}".strip()
        if line and draw.textlength(trial,font=face)>width:
            draw.text((x,y),line,font=face,fill=color)
            line=word
            y+=size+8
        else:
            line=trial
    if line:
        draw.text((x,y),line,font=face,fill=color)
        y+=size+8
    return y


def picture(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",",1)[1]))).convert("RGB")


def panel(row):
    canvas=Image.new("RGB",(1540,1200),BG)
    draw=ImageDraw.Draw(canvas)
    draw.text((38,18),row["title"]+" · "+row["wall"],font=font(44,True),fill=TEXT)
    y=draw_wrap(draw,(38,80),row["proposal"],1460,29)
    context=picture(row["context"])
    context.thumbnail((1120,280),Image.Resampling.LANCZOS)
    canvas.paste(context,((canvas.width-context.width)//2,y+9))
    y+=context.height+25
    options=row["options"]
    captions=["A · Shared-corner grid","B · Original atlas grid","Gold · Recorded visible edges"]
    imgs=[picture(options[0]["images"][0]),picture(options[1]["images"][0]),picture(row["gold"])]
    for i,(cap,img) in enumerate(zip(captions,imgs,strict=True)):
        x=37+490*i
        draw.text((x,y),cap,font=font(27,True),fill=TEXT)
        img.thumbnail((466,470),Image.Resampling.LANCZOS)
        canvas.paste(img,(x+(466-img.width)//2,y+48))
    y+=max(im.height for im in imgs)+64
    a,b=(o["score"] for o in options)
    y=draw_wrap(draw,(38,y),f"Whole export · {a[0]} → {b[0]} correct · {a[1]} → {b[1]} extras · {a[2]} → {b[2]} exact position and material matches.",1460,27,MUTED)
    y=draw_wrap(draw,(38,y+6),"Which grid has the right block size and covers the visible wall? Mark any edge that should keep the two adjoining faces aligned.",1460,28,GOLD,True)
    assert y+27 < canvas.height
    return canvas.crop((0,0,canvas.width,y+27))


def main(rows):
    cases=[]
    for i,row in enumerate(rows,1):
        rel=Path("sky")/row["id"]/"charts.json"
        spec=(row["wall"],row["panel"],row["id"],row["wall"],f"O{i} · {row['title']}",row["note"])
        charts=[option(WORKTREES/Path(root),rel,row["wall"]) for root in row["roots"]]
        cases.append(make(spec,dict(charts=charts,scores=row["scores"])))
    template=Path(__file__).with_name("floating_grid_signoff.html").read_text(encoding="utf-8")
    template=template.replace("floating-grid-signoff-02","floating-corner-overwrite-review-01")
    template=template.replace("floating-grid-signoff-feedback.json","floating-corner-overwrite-feedback.json")
    template=template.replace("A · Current grid","A · Shared-corner grid").replace("B · Tested alternative","B · Original atlas grid")
    template=template.replace("My proposed adjustment","What changed at assembly")
    REVIEW.mkdir(parents=True,exist_ok=True)
    (REVIEW/"review.html").write_text(template.replace("__PAYLOAD__",json.dumps(cases)),encoding="utf-8")
    panels=[panel(row) for row in cases]
    mobile=Image.new("RGB",(1540,143+sum(p.height+12 for p in panels)),BG)
    draw=ImageDraw.Draw(mobile)
    draw.text((38,22),"Floating grids · corner overwrite review",font=font(51,True),fill=TEXT)
    draw_wrap(draw,(38,90),"Reply with the case ID and what to keep. Check coverage, alignment and block size independently.",1460,28,MUTED)
    y=143
    for pic in panels:
        draw.line((38,y,1502,y),fill="#567080",width=2)
        mobile.paste(pic,(0,y+7))
        y+=pic.height+12
    mobile.save(REVIEW/"mobile-review.png",optimize=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases",type=Path,nargs="?",default=REVIEW/"queue.json")
    main(json.loads(parser.parse_args().cases.read_text(encoding="utf-8")))
