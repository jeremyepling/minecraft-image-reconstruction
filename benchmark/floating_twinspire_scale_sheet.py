"""Draw the Twinspire ruler transfer on the original image and two matching native crops."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from benchmark.floating_grid_signoff import WORKTREES, project, read
from benchmark.floating_extra_budget import paths

CASE = "test-twinspire2-earth-high-right"
EXISTING = WORKTREES/"floating-occlusion-fragments/runs/occlusion-fragments/original"
DEST = Path(__file__).parent/"reviews/floating-grid-round-04"
LEFT = (513,365,729,677)
RIGHT = (866,321,1041,790)
UNIT = np.array([(0,0),(1,0),(1,1),(0,1),(0,0)])
COLORS = dict(bg="#14202b",white="#eff4f7",muted="#c4d4e0",cyan="#40eef7",gold="#ffcf58")


def font(size,bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf",size)


def grid(rgb, chart, box, scale):
    picture = rgb.crop(box)
    size = tuple(round(v*scale) for v in picture.size)
    picture = picture.resize(size,Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(picture)
    h = np.asarray(chart["H"],float)
    for cell in chart.get("export_cells",chart["cells"]):
        uv = UNIT+[cell["u"],cell["v"]]
        xy = scale*(project(h,uv)-np.array(box[:2]))
        draw.line([tuple(p) for p in xy],fill=COLORS["cyan"],width=2)
    return picture


def native_grid(rgb, frame, surface, box, scale):
    # Native ownership clips gold boundaries at occlusion instead of showing hidden sides.
    with np.load(frame/"labels.npz") as labels:
        ids = labels["face_id"]
    faces = read(frame/"labels.json")["faces"]
    needed = [f["id"] for f in faces if f["surface_id"]==surface]
    line = np.zeros_like(ids,dtype=bool)
    line[1:] |= ids[1:] != ids[:-1]
    line[:,1:] |= ids[:,1:] != ids[:,:-1]
    line &= np.isin(ids,needed)
    alpha = Image.fromarray(line.astype("uint8")*255).crop(box)
    rgb = rgb.crop(box)
    size = tuple(round(n*scale) for n in rgb.size)
    rgb = rgb.resize(size,Image.Resampling.LANCZOS)
    rgb.paste(COLORS["gold"],(0,0,*size),alpha.resize(size,Image.Resampling.NEAREST))
    return rgb


def paste_center(canvas,picture,x,y,width):
    canvas.paste(picture,(x+(width-picture.width)//2,y))


def load_chart(path,id):
    return next(c for c in read(path)["charts"] if c["id"]==id)


def render(candidate=None,new_count=None):
    case=next(c for c in paths("original")[0][3] if c["id"]==CASE)
    frame=Path(case["frame"])
    with Image.open(case["variants"]["sky"]["image"]) as im:
        rgb=im.convert("RGB")
    before=EXISTING/"charts"/"sky"/CASE/"charts.json"
    scale=1.45
    chart_left=load_chart(before,"mask-2")
    chart_right=load_chart(before,"mask-1")
    context=rgb.crop((385,184,1082,794))
    ctxdraw=ImageDraw.Draw(context)
    for box,label in [(LEFT,"Left wall L"),(RIGHT,"Right wall R")]:
        coord=(box[0]-385,box[1]-184,box[2]-385,box[3]-184)
        ctxdraw.rectangle(coord,outline=COLORS["cyan"],width=3)
        ctxdraw.text((coord[0]+7,coord[1]+5),label,font=font(18,True),fill=COLORS["cyan"],stroke_width=2,stroke_fill="#172a33")
    cells=[
        ("L · Measured grid","8 columns · 8 rows before occlusion",grid(rgb,chart_left,LEFT,scale)),
        ("R · Original measured grid","13 columns · up to 12 rows · 143 cells",grid(rgb,chart_right,RIGHT,scale)),
        ("R · Recorded visible grid","8 columns · 10 rows · 80 faces",native_grid(rgb,frame,16,RIGHT,scale)),
    ]
    if candidate:
        chart=load_chart(candidate,"mask-1")
        cells.insert(2,("R · After ruler transfer",new_count or "Experimental grid",grid(rgb,chart,RIGHT,scale)))
    width=433
    top=825
    canvas=Image.new("RGB",(width*len(cells)+30,top+755),COLORS["bg"])
    d=ImageDraw.Draw(canvas)
    d.text((35,23),"Twinspire · one block on the left and right",font=font(45,True),fill=COLORS["white"])
    d.text((35,83),"Both walls face the same way. They sit at different depths.",font=font(27),fill=COLORS["muted"])
    d.text((35,120),"Cyan outlines measured cells. Gold traces recorded visible block edges.",font=font(27),fill=COLORS["muted"])
    paste_center(canvas,context,0,163,canvas.width)
    for i,(label,description,pic) in enumerate(cells):
        x=15+i*width
        d.text((x+18,top-38),label,font=font(26,True),fill=COLORS["white"])
        d.text((x+18,top-4),description,font=font(21),fill=COLORS["muted"])
        paste_center(canvas,pic,x,top+34,width)
    path=DEST/("twinspire-scale-test.png" if candidate else "twinspire-scale-before.png")
    canvas.save(path,optimize=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--count")
    args=parser.parse_args()
    render(args.candidate,args.count)
