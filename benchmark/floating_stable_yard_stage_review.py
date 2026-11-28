"""Draw Stable Yard source-depth masks and the atlas-ruler placement trace."""
from argparse import ArgumentParser
from colorsys import hsv_to_rgb
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import digest, read
from benchmark.floating_corner_overwrite_review import BG, GOLD, MUTED, TEXT, draw_wrap, font
from benchmark.floating_extra_budget import paths
from benchmark.floating_grid_signoff import WORKTREES
from engine.geometry import UNIT, project
from engine.surface_perimeter import _native_alpha

ARCHIVE = WORKTREES/"floating-extra-isolation/runs/extra-isolation/holdout3/large-contraction"
PEER = WORKTREES/"floating-peer-ruler-refit/runs/peer-ruler-refit/atlas-first/holdout3"
PRIOR_PARTS = WORKTREES/"floating-mixed-wall-refit/runs/mixed-refit/holdout3/source"
PANEL = "holdout3"
EDGE = WORKTREES/"floating-edge-aligned-ruler/runs/edge-aligned-ruler/holdout3"
CYAN = "#47f0fc"
PINK = "#fb75ad"
VIOLET = "#a294ff"
ORANGE = "#ff894c"



def region_colors(ids):
    steps = float(np.sqrt(5)-1)/2
    palette = np.array([np.asarray(hsv_to_rgb((k*steps+.91)%1, .55+.13*(k%2), .94))*255
                        for k in range(int(ids.max())+1)])
    return palette[ids]


def color_overlay(rgb, mask, colors, strength=.62):
    gray = np.asarray(Image.fromarray(rgb).convert("L"),dtype=float)
    backdrop = np.uint8(rgb*.48 + gray[:,:,None]*.12)
    backdrop[mask] = np.uint8((1-strength)*rgb[mask] + strength*colors)
    return Image.fromarray(backdrop)


def source_masks(rgb, mask, folder, case, frame, new_parts=None):
    group = Path(case["dataset"])/case["id"]
    field = folder.parent/"fields"/group/"depth.npz"
    for stamp in (field.parent/"receipt.json",field.parent.parent/"receipt.json"):
        if stamp.is_file():
            rec = read(stamp)
            assert rec["source_sha256"] == case["variants"]["sky"]["source_sha256"]
            if "fields_sha256" in rec:
                assert rec["fields_sha256"]["depth"] == digest(field)
            break
    with np.load(field) as bundle:
        d = bundle["value"].squeeze()
    d = cv2.resize(d.astype(np.float32),rgb.shape[1::-1])
    yy,xx = np.where(mask)
    system = np.column_stack((np.ones(len(yy)),xx-xx.mean(),yy-yy.mean()))
    residual = d[mask] - system@np.linalg.lstsq(system[::11],d[mask][::11],rcond=None)[0]
    scale = np.percentile(abs(residual),90)
    alpha = .5+.5*np.clip(residual/max(scale,1e-9),-1,1)
    nearer = np.array([243.,100.,166.])
    farther = np.array([116.,98.,238.])
    signed = color_overlay(rgb,mask,alpha[:,None]*nearer+(1-alpha[:,None])*farther)

    folder_old = PRIOR_PARTS/case["id"]/"region-ids.png"
    source_path = (new_parts/case["id"]/"mask-0-ids.png") if new_parts else folder_old
    source_heading = "Depth-first regions" if source_path.is_file() and new_parts else "Earlier source regions"
    if not source_path.is_file():
        source_path = folder_old
    with Image.open(source_path) as im:
        source_ids = np.asarray(im)
    assert source_ids.shape == mask.shape
    source_support = source_ids > 0
    assert not np.any(source_support & ~mask)
    source = color_overlay(rgb,source_support,region_colors(source_ids[source_support]))

    with np.load(frame/"labels.npz") as bundle:
        owners = bundle["face_id"]
    labels = read(frame/"labels.json")
    groups = {}
    for face in labels["faces"]:
        normal = tuple(face["normal_world"])
        location = round(float(np.dot(face["corners_world"][0],normal)),5)
        groups[face["id"]] = (normal,location)
    group_ids = {group:index+1 for index,group in enumerate(sorted(set(groups.values())))}
    lut = np.zeros(owners.max()+1,np.int32)
    for face, group in groups.items():
        lut[face] = group_ids[group]
    ref_ids = lut[owners]
    recorded = mask & (ref_ids > 0)
    ref = color_overlay(rgb,recorded,region_colors(ref_ids[recorded]))
    edge = cv2.Canny(np.uint8(mask)*255,30,90)>0
    initial = rgb.copy()
    initial[edge] = (255,201,67)
    return [(Image.fromarray(initial),"Original mask","Gold outlines one input mask that spans several surfaces."),
            (signed,"Source depth residual","Pink is nearer, violet farther after subtracting the wall's estimated slope. This shows relative depth evidence."),
            (source,source_heading,"Colors distinguish proposed regions, independently of the reference panel."),
            (ref,"Recorded visible planes","Colors distinguish recorded physical planes within the same mask. Reference used for checking.")]


def ruler_views(rgb, mask, case, frame, trace):
    ident = case["id"]
    rel = Path("sky")/ident/"charts.json"
    charts = [next(c for c in read(root/"charts"/rel)["charts"] if c["id"]=="mask-0")
              for root in (ARCHIVE,PEER/"relaxed")]
    info = read(PEER/"audit"/(ident+".json.gz"))["charts"]["mask-0"]
    audit = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(audit)
    x = np.where(mask)[1]+.5
    points = np.quantile(x,[.4,.6])
    for pos in points:
        d.line((pos,0,pos,rgb.shape[0]),fill="#c9d1ff",width=2)
    h = np.asarray(info["homography"])
    anchor = project(h,UNIT)
    d.line([*map(tuple,anchor),tuple(anchor[0])],fill="#6bffa4",width=4)
    for candidate in info["far"]:
        x0,y0 = project(h,np.array(candidate["uv"])+.5)
        if np.isfinite((x0,y0)).all():
            d.ellipse((x0-2,y0-2,x0+2,y0+2),fill=ORANGE)
    fit_note = f'Green is the fitting tile ({info["near"]["material"].split(":")[-1].replace("_"," ")}). Orange marks the distant checks that failed.'

    options = [draw_grid(rgb, chart) for chart in charts]
    with np.load(frame/"labels.npz") as bundle:
        owners = bundle["face_id"]
    faces = [f["id"] for b in trace["lost"] for f in b["faces"]]
    lost = np.isin(owners,faces)
    painted = color_overlay(rgb,lost,np.tile([251,103,140],(lost.sum(),1)),.72)
    pen = ImageDraw.Draw(painted)
    sx,sy = np.where(lost)
    if len(sx):
        pen.rectangle((sy.min()-3,sx.min()-3,sy.max()+3,sx.max()+3),outline=PINK,width=2)
    return [(options[0],"Original grid",f'{len(charts[0]["cells"]):,} cells proposed on one wall mask.'),
            (audit,"Fit and distant check",fit_note),
            (options[1],"Forced refit",f'{len(charts[1]["cells"]):,} cells. Near and far checks did not establish the same ruler.'),
            (painted,"Eight lost placements","Pink covers the recorded visible faces of the eight lost blocks. The overlay uses the reference.")]


def draw_grid(rgb, chart):
    image = Image.fromarray(rgb.copy())
    pen = ImageDraw.Draw(image)
    h = np.asarray(chart["H"])
    for c in chart["cells"]:
        xy = project(h,UNIT+[c["u"],c["v"]])
        if np.isfinite(xy).all():
            pen.line([*map(tuple,xy),tuple(xy[0])],fill=CYAN,width=1)
    return image


def edge_views(rgb, case, frame, variant):
    name = case["id"]
    row = read(EDGE/"rows"/(name+".json.gz"))
    line = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(line)
    for family,color in zip(row["charts"]["mask-0"]["lines"],(CYAN,PINK),strict=True):
        for accepted,seg in zip(family.get("inliers",[]),family.get("lines",[]),strict=True):
            if accepted:
                d.line([tuple(p) for p in seg["ends"]],fill=color,width=3)
    rel = Path("sky")/name/"charts.json"
    old,new = [next(c for c in read(root/"charts"/rel)["charts"] if c["id"]=="mask-0")
               for root in (PEER/"relaxed",EDGE/variant)]
    with np.load(frame/"labels.npz") as bundle:
        owners = bundle["face_id"]
    borders = (cv2.Canny(np.uint8(owners>0)*255,30,90)>0)
    borders[1:] |= owners[1:] != owners[:-1]
    borders[:,1:] |= owners[:,1:] != owners[:,:-1]
    gold = rgb.copy()
    gold[borders & (owners>0)] = (255,195,60)

    def totals(key):
        p = row["score"][key]
        g = p["geometry"]["strict"]
        return f'{g["correct"]} correct, {g["extras"]} extras, {p["scores"]["strict"]["correct"]} exact material matches.'

    return [(draw_grid(rgb,old),"Previous atlas ruler",totals("peer")),
            (line,"Structural lines used","Cyan and pink are the two direction families. Several widely separated lines must agree."),
            (draw_grid(rgb,new),"Edge-aligned refit","Forced for diagnosis. "+totals(variant)),
            (Image.fromarray(gold),"Recorded visible boundaries","Gold traces the independently recorded visible faces.")]


def render(out, mode, new_parts=None, edge_variant="refit-material"):
    folder,_,_,fixtures = paths(PANEL)[0]
    cohort = {c["id"]:c for c in fixtures}
    trace_rows = {r["id"]:r for r in read(Path(__file__).parent/"reviews/floating-stable-yard-depth-review-01/forced-native-trace.json")["views"]} if mode=="ruler" else {}
    width=1540
    page=Image.new("RGB",(width,5400),BG)
    d=ImageDraw.Draw(page)
    title = {"depth":"Stable Yard · split depth before fitting",
             "ruler":"Stable Yard · trace the ruler and export",
             "edges":"Stable Yard · align the ruler to visible edges"}[mode]
    d.text((38,22),title,fill=TEXT,font=font(48,True))
    y=100
    for side,title in (("low-left","Low-left view"),("high-right","High-right view")):
        name=f"train-stable-yard-1-warm-{side}"
        case=cohort[name]
        frame=Path(case["frame"])
        with Image.open(case["variants"]["sky"]["image"]) as im:
            rgb=np.asarray(im.convert("RGB"))
        source=read(folder/"sky"/name/"source-handoff/surfaces.json")
        plane=next(p for p in source["planes"] if p["id"]=="mask-0")
        alpha,(x,y0,x1,y1)=_native_alpha(plane["support"],rgb.shape[1::-1])
        native=np.zeros(rgb.shape[:2],bool)
        native[y0:y1,x:x1]=np.asarray(alpha)>0
        crop=(max(x-22,0),max(y0-30,0),min(x1+22,rgb.shape[1]),min(y1+30,rgb.shape[0]))
        d.line((38,y,width-38,y),fill="#586c7b",width=2)
        d.text((38,y+18),title,fill=TEXT,font=font(38,True))
        y+=73
        rec=trace_rows.get(name)
        if mode=="depth":
            panels=source_masks(rgb,native,folder,case,frame,new_parts)
        elif mode=="ruler":
            panels=ruler_views(rgb,native,case,frame,rec)
        else:
            panels=edge_views(rgb,case,frame,edge_variant)
        for row in range(2):
            height=0
            for col,(im,label,note) in enumerate(panels[row*2:row*2+2]):
                xslot=38+736*col
                d.text((xslot,y),label,fill=TEXT,font=font(27,True))
                im=im.crop(crop)
                im.thumbnail((700,465),Image.Resampling.LANCZOS)
                page.paste(im,(xslot,y+45))
                end=draw_wrap(d,(xslot,y+im.height+54),note,680,24,MUTED)
                height=max(height,end-y)
            y+=height+22
    assert y<page.height
    page.crop((0,0,width,y+20)).save(out,optimize=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("depth","ruler","edges"))
    parser.add_argument("out",type=Path)
    parser.add_argument("--new-parts",type=Path)
    parser.add_argument("--edge-variant",default="refit-material")
    args=parser.parse_args()
    render(args.out,args.mode,args.new_parts,args.edge_variant)
