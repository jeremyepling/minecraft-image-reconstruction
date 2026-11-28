"""Archive the part-depth results and draw paired source and reference offsets."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from benchmark.floating_part_depth import LOCATION, OUT, PANELS, REVIEW, WORKTREES, paths, read, save
from benchmark.floating_bridges_report import read_arm
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

EXAMPLES=[
    ("holdout4","train-riverside-mill-1-warm-low-left","mask-0","Mill, low left"),
    ("holdout4","train-riverside-mill-1-warm-high-right","mask-0","Mill, high right"),
    ("holdout2","train-shrine-02-ochre-low-left","mask-0-wall-0","Shrine, low left"),
    ("holdout4","train-viaduct-01-riverstone-low-left","mask-0","Viaduct, low left"),
    ("original","test-castle-01-ashlar-high-right","mask-0","Castle, high right"),
    ("holdout2","train-steppedhall2-cool-high-right","mask-0-wall-0","Stepped Hall, high right"),
]
UNIT=np.array([(0,0),(1,0),(1,1),(0,1),(0,0)])
COLORS={-1:(60,184,255),0:(174,183,190),1:(255,187,50)}


def totals(rows):
    result={}
    for arm in rows[0]["score"]:
        pooled={}
        for field in ("geometry","scores"):
            for kind in ("strict","facade"):
                counts={k:sum(r["score"][arm][field][kind][k] for r in rows) for k in ("correct","missed","extras")}
                counts["f1"]=2*counts["correct"]/(2*counts["correct"]+counts["missed"]+counts["extras"])
                pooled[field+"_"+kind]=counts
        pooled["paired"]={k:sum(r["paired"][arm][k] for r in rows) for k in rows[0]["paired"][arm]}
        result[arm]=pooled
    return result


def example(panel,name,cid,title,row):
    (source,_,_,cases),_,_=paths(panel)
    case=next(c for c in cases if c["id"]==name)
    with Image.open(case["variants"]["sky"]["image"]) as im:
        rgb=im.convert("RGB")
    baseline=WORKTREES/"floating-extra-isolation/runs/extra-isolation"/panel/"large-contraction"
    chart=next(c for c in read(baseline/"charts/sky"/name/"charts.json")["charts"] if c["id"]==cid)
    plane=next(p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==cid)
    alpha,box=_native_alpha(plane["support"],rgb.size)
    mask=Image.new("L",rgb.size)
    mask.paste(alpha,box[:2])
    back=Image.composite(rgb,rgb.point(lambda x:int(x*.38)),mask)
    x,y,r,b=box
    crop=(max(0,x-12),max(0,y-12),min(rgb.width,r+12),min(rgb.height,b+12))
    detail=next(c for c in row["charts"] if c["chart"]==cid)
    font=ImageFont.truetype("arial.ttf",18)
    small=ImageFont.truetype("arial.ttf",15)
    out=REVIEW/"examples"/name
    out.mkdir(parents=True,exist_ok=True)
    context=rgb.copy()
    ImageDraw.Draw(context).rectangle(crop,outline="#ffc347",width=3)
    context.thumbnail((960,540))
    context.save(out/"context.jpg",quality=90)
    rgb.crop(crop).save(out/"source-crop.jpg",quality=93)
    back.crop(crop).save(out/"mask.jpg",quality=93)
    mapped={}
    for arm,label,key in (("large-contraction","Flat grid",None),("source","Source depth estimate","source_offsets"),
                          ("visible-reference","Visible reference depth","reference_offsets")):
        offsets={(u,v):d for u,v,d in detail[key]} if key else {}
        pic=back.copy().convert("RGBA")
        overlay=Image.new("RGBA",pic.size)
        draw=ImageDraw.Draw(overlay)
        for cell in chart["cells"]:
            uv=(cell["u"],cell["v"])
            pts=list(map(tuple,project(np.asarray(chart["H"]),UNIT+uv)))
            d=offsets.get(uv,0)
            color=COLORS[d]
            draw.polygon(pts,fill=(*color,100 if d else 4))
            draw.line(pts,fill=(*color,240 if d else 145),width=2 if d else 1)
        view=Image.alpha_composite(pic,overlay).crop(crop).convert("RGB")
        view.thumbnail((780,470),Image.Resampling.LANCZOS)
        canvas=Image.new("RGB",(800,560),(19,27,36))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f"{title}  |  {label}",font=font,fill="white")
        canvas.paste(view,((800-view.width)//2,42+(470-view.height)//2))
        x=12
        for d,text in ((1,"1 block forward"),(-1,"1 block back"),(0,"unchanged")):
            draw.rectangle((x,521,x+14,535),fill=COLORS[d])
            draw.text((x+21,518),text,font=small,fill="white")
            x+=170
        draw.text((525,518),f"{len(offsets)} / {len(chart['cells'])} cells moved",font=small,fill="white")
        canvas.save(out/(arm+".jpg"),quality=93)
        mapped[arm]=str((out/(arm+".jpg")).relative_to(REVIEW)).replace("\\","/")
    source_pack=OUT/panel/"source/assembly"/LOCATION/name
    _,old=read_arm(baseline/"assembly"/LOCATION/name)
    _,new=read_arm(source_pack) if source_pack.is_dir() else (None,old)
    frame=Path(case["frame"])
    with np.load(frame/"labels.npz") as arrays:
        owners=arrays["face_id"]
    faces=read(frame/"labels.json")["faces"]
    painted=np.asarray(rgb).copy()
    for coords,color in ((new.keys()-old.keys(),(49,229,153)),(old.keys()-new.keys(),(251,101,104))):
        ids=[f["id"] for f in faces if tuple(f["xyz_world"]) in coords]
        where=np.isin(owners,ids)
        painted[where]=.35*painted[where]+.65*np.array(color)
    outcome=Image.fromarray(painted).crop(crop)
    outcome.thumbnail((780,470),Image.Resampling.LANCZOS)
    canvas=Image.new("RGB",(800,560),(19,27,36))
    draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f"{title}  |  Change in correctly placed visible blocks",font=font,fill="white")
    canvas.paste(outcome,((800-outcome.width)//2,42+(470-outcome.height)//2))
    paired=row["paired"]["source"]
    draw.text((12,518),f"Green   {paired['gained']} gained",font=small,fill=(49,229,153))
    draw.text((175,518),f"Red   {paired['lost']} lost",font=small,fill=(251,101,104))
    draw.text((315,518),f"Whole-build extras   +{paired['added']} / −{paired['removed']}",font=small,fill="white")
    canvas.save(out/"placement-changes.jpg",quality=93)
    mapped["placement-changes"]=str((out/"placement-changes.jpg").relative_to(REVIEW)).replace("\\","/")
    return dict(id=name,panel=panel,chart=cid,title=title,crop=crop,files=mapped,
                source_estimate=detail["source"],source_offsets=detail["source_offsets"],
                reference_offsets=detail["reference_offsets"],score=row["score"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active",type=Path,default=WORKTREES/"floating-textured-cohort/benchmark/reviews/floating-textured-cohort-01/scores.json.gz",
                   help="Frozen active-cohort audit with panel, frame, source hash and active flag")
    a=p.parse_args()
    rows=[read(path) for panel in PANELS for path in sorted((OUT/panel/"cases").glob("*.json.gz"))]
    assert len(rows)==len({(r["panel"],r["id"]) for r in rows})==92
    audit=read(WORKTREES/"floating-feedback-audit/benchmark/reviews/floating-feedback-audit-01/decoded.json.gz")["rows"]
    companion={(r["panel"],r["id"]):r for r in audit}
    for row in rows:
        archived=companion[row["panel"],row["id"]]
        assert archived["source_sha256"]==row["source_sha256"]
        assert archived["scores"]["large-contraction"]["geometry"]==row["score"]["large-contraction"]["geometry"]
        row["layout"]=archived["layout"]
    reviewed={r["layout"] for r in rows if any(key in r["id"] for key in ("riverside-mill-1","shrine-02-ochre"))}
    groups={"all92":rows,"sky":[r for r in rows if r["panel"]=="sky"],
            "mill-and-shrine":[r for r in rows if r["layout"] in reviewed],
            "other-layouts":[r for r in rows if r["layout"] not in reviewed]}
    cohort=read(a.active)
    selection={(r["panel"],r["id"]):r for r in cohort["rows"]}
    assert all(selection[r["panel"],r["id"]]["source_sha256"]==r["source_sha256"] for r in rows)
    active=[r for r in rows if selection[r["panel"],r["id"]]["active"]]
    groups.update(active=active,active_other=[r for r in active if r["layout"] not in reviewed])
    for r in rows:
        r["active"]=selection[r["panel"],r["id"]]["active"]
    packed={k:dict(count=len(g),totals=totals(g)) for k,g in groups.items()}
    save(REVIEW/"summary.json",dict(packed,cohort=cohort["cohort"],cohort_manifest_lf_sha256=cohort["cohort_manifest_lf_sha256"]))
    save(REVIEW/"conditions.json.gz",rows)
    for group,result in packed.items():
        for arm,stats in result["totals"].items():
            print(group,arm,[round(100*stats[k]["f1"],3) for k in ("geometry_strict","geometry_facade","scores_strict","scores_facade")],
                  stats["geometry_strict"],stats["paired"])
    data={(r["panel"],r["id"]):r for r in rows}
    examples=[example(g,n,c,t,data[g,n]) for g,n,c,t in EXAMPLES]
    save(REVIEW/"examples.json",examples)


if __name__=="__main__":
    main()
