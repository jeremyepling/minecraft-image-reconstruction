"""Prepare measured grid reviews with separate extent, phase, size and depth evidence."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmark.floating_extra_feedback import picture, read, paths, project, _native_alpha, WORKTREES
from benchmark.floating_feedback_viewer import ISOLATION, SCALE
from benchmark.floating_parallel_depth_audit import OUT, faces_at_depth
from benchmark.floating_visible_row import RUN

PART=WORKTREES/"floating-part-grids/runs/part-depth"
CROSS=WORKTREES/"floating-cross-ruler/runs/cross-ruler"
CASES=[
    ("row","original","test-twinspire2-earth-high-right","mask-2","Twinspire · upper brick row",
     "Keep the horizontal spacing and test one more row of visible brick above the current grid. Stop at the projecting pale blocks. The red brick pattern repeats inside each full block."),
    ("depth","holdout4","train-riverside-mill-1-warm-low-left","mask-0","Riverside Mill · window and porch",
     "Keep the wall grid and separate the recessed and projecting parts by one block in depth. Preserve the window cross where the source shows it. A change in material alone cannot establish depth."),
    ("depth","holdout4","train-viaduct-01-riverstone-low-left","mask-0","Viaduct · projecting faces",
     "Retain the sampled cells and test a separate one-block depth for the coherent projecting region. Inspect the return faces where the two planes meet."),
    ("depth","original","test-castle-01-ashlar-high-right","mask-0","Castle · harmful depth adjustment",
     "Keep the existing placement here. The predicted depth step moved cells and lost correct positions. Check which visible rim, if any, actually supports a step."),
    ("depth","holdout2","train-shrine-02-ochre-low-left","mask-0-wall-0","Shrine · stairs and right wall",
     "Keep the right wall and give the visible stair treads and risers separate small surfaces. The tested one-block depth rule leaves this example unresolved."),
    ("cross","sky","train-pillars-01-marble-low-left","mask-1","Pillars · ruler from another wall",
     "Use the visible full faces on neighboring surfaces to rule out the microscopic grid. B reduces extras but also loses some correct positions. Check which edges justify further extension."),
    ("origin","sky","validation-hillside-chapel-1-warm-low-left","mask-0","Chapel · left edge",
     "Retain the larger full-block spacing and adjust the origin at the left edge. B is a reference-assisted shift; check whether the far end drifts before approving it."),
]
SQUARE=np.array([[0,0],[1,0],[1,1],[0,1]])


def get_chart(folder,rel,wall):
    return next(c for c in read(folder/"charts"/rel)["charts"] if c["id"]==wall)


def paint(image,mask,color,amount):
    image[mask]=np.uint8((1-amount)*image[mask]+amount*np.array(color))


def render(chart,rgb,owners,plane_ids,faces,case_mask,gold,box,shifts):
    x,y,r,b=box
    covered=np.zeros(owners.shape,"uint8")
    lines=np.zeros_like(covered)
    signed=np.zeros(owners.shape,"int8")
    h=np.asarray(chart["H"])
    for cell in chart["cells"]:
        u,v=cell["u"],cell["v"]
        pts=project(h,SQUARE+[u,v])
        if not np.isfinite(pts).all() or np.max(abs(pts))>1e6:
            continue
        pts=np.rint(pts).astype("int32")
        cv2.fillConvexPoly(covered,pts,1)
        cv2.polylines(lines,[pts],True,1,1)
        if (u,v) in shifts:
            cv2.fillConvexPoly(signed,pts,shifts[u,v])
    current=rgb.copy()
    paint(current,signed>0,(243,111,191),.5)
    paint(current,signed<0,(130,121,255),.55)
    wire=lines.astype(bool)
    current[wire]=(0,249,255)
    combined=current.copy()
    combined[gold]=(255,199,40)
    combined[gold & wire]=(255,255,255)
    target=np.isin(owners,plane_ids)
    area_target=target[y:b,x:r]
    coverage=(covered[y:b,x:r].astype(bool) & area_target).sum()/max(1,area_target.sum())
    inv=np.linalg.inv(h)
    corner_sets=[project(inv,np.asarray(f["corners_px"])) for f in faces]
    metrics={"covered":round(100*coverage,1)}
    if corner_sets:
        uv=np.asarray(corner_sets)
        a,bcoord=uv.transpose(2,0,1)
        area=.5*abs((a*np.roll(bcoord,-1,axis=1)-bcoord*np.roll(a,-1,axis=1)).sum(axis=1))
        metrics.update(cells=round(float(np.median(area)),2),phase=round(float(np.median(abs(uv-uv.round()))),2),faces=len(uv))
    # Grid wire is drawn independently from target ownership. Gold is visible reference only.
    return [picture(Image.fromarray(pic[y:b,x:r])) for pic in (current,combined)],metrics


def make(spec):
    kind,panel,name,wall,title,proposal=spec
    (root,_,_,cases),_,_=paths(panel)
    case=next(c for c in cases if c["id"]==name)
    frame=Path(case["frame"])
    labels=read(frame/"labels.json")
    faces=labels["faces"]
    by_id=faces_at_depth(labels)
    with np.load(frame/"labels.npz") as data:
        ids=data["face_id"]
    with Image.open(case["variants"]["sky"]["image"]) as im:
        rgb=np.array(im.convert("RGB"))
    plane=next(p for p in read(root/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
    alpha,(x,y,r,b)=_native_alpha(plane["support"],ids.shape[::-1])
    mask=np.zeros_like(ids,bool)
    mask[y:b,x:r]=np.asarray(alpha)>0
    box=(max(x-10,0),max(y-10,0),min(r+10,ids.shape[1]),min(b+10,ids.shape[0]))
    x,y,r,b=box
    keys,weights=np.unique(ids[mask],return_counts=True)
    areas=Counter()
    counts=dict(zip(keys,weights))
    for k,n in counts.items():
        if k in by_id:
            areas[by_id[k]]+=int(n)
    principal=areas.most_common(1)[0][0]
    plane_ids=[i for i in by_id if by_id[i]==principal]
    eligible=[f for f in faces if f["id"] in plane_ids and counts.get(f["id"],0)>=.5*f["visible_pixels"]
              and all(f["corner_visible"]) and np.allclose(np.linalg.norm(np.roll(f["corners_world"],-1,axis=0)-np.array(f["corners_world"]),axis=1),1.)]
    edges=np.zeros_like(ids,bool)
    edges[1:] |= ids[1:]!=ids[:-1]
    edges[:,1:] |= ids[:,1:]!=ids[:,:-1]
    gold=edges & (ids!=0)
    truth=rgb.copy()
    truth[gold]=(255,199,40)
    depth_truth=rgb.copy()
    for direction,distance in areas:
        if direction!=principal[0] or np.isclose(distance,principal[1]):
            continue
        layer=np.isin(ids,[fid for fid,p in by_id.items() if p==(direction,distance)]) & mask
        paint(depth_truth,layer,(243,111,191) if distance>principal[1] else (130,121,255),.58)
    depth_truth[gold]=(255,199,40)
    rel=Path("sky")/name/"charts.json"
    a=ISOLATION/panel/"large-contraction"
    shifts={}
    if kind=="row":
        a=SCALE/panel
        b_folder=RUN/panel/"row-with-material"
        scores=read(OUT/"top-row.json.gz")[name]["stats"]["score"]
        after=scores["row-with-material"];before=scores["previous"]
    elif kind=="cross":
        b_folder=CROSS/panel
        rows=read(WORKTREES/"floating-cross-ruler/benchmark/reviews/floating-cross-ruler-01/conditions.json.gz")
        scores=next(r for r in rows if r["group"]==panel and r["id"]==name)["score"]
        before=scores["control"];after=scores["candidate"]
    elif kind=="origin":
        a=Path("runs/feedback-boundary")/panel/"resample"
        b_folder=a.parent/"origin"
        scores=read(OUT.parent/"floating-wall-feedback-01/boundary-and-origin.json.gz")[name]["scores"]["score"]
        before=scores["resample"];after=scores["origin"]
    else:
        data=read(PART/panel/"cases"/(name+".json.gz"))
        rec=next(c for c in data["charts"] if c["chart"]==wall)
        shifts={(u,v):o for u,v,o in rec["source_offsets"]}
        b_folder=a
        before=data["score"]["large-contraction"];after=data["score"]["source"]
    options=[]
    for folder,shift,score in [(a,{},before),(b_folder,shifts,after)]:
        imgs,m=render(get_chart(folder,rel,wall),rgb,ids,plane_ids,eligible,mask,gold,box,shift)
        geom=score["geometry"]["strict"]
        options.append(dict(images=imgs,metrics=m,score=[geom["correct"],geom["extras"],score["scores"]["strict"]["correct"]]))
    context=Image.fromarray(rgb)
    ImageDraw.Draw(context).rectangle(box,outline="gold",width=4)
    context.thumbnail((720,360))
    return dict(id=name+"-"+kind,title=title,wall=wall,kind=kind,proposal=proposal,context=picture(context),
                rgb=picture(Image.fromarray(rgb[y:b,x:r])),gold=picture(Image.fromarray(truth[y:b,x:r])),
                depth_ref=picture(Image.fromarray(depth_truth[y:b,x:r])),options=options, moved=len(shifts))


if __name__=="__main__":
    cases=[make(s) for s in CASES]
    template=Path(__file__).with_name("floating_grid_signoff.html").read_text(encoding="utf-8")
    (OUT/"review.html").write_text(template.replace("__PAYLOAD__",json.dumps(cases)),encoding="utf-8")
