"""Summarize actual decoded blocks for each floating bridge and sky-only replay."""
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from benchmark.expanded_baseline import read,save
from benchmark.mask_blocks import decoded_pack
from benchmark.floating_horizontal_bridges import OUT,OLD,FRESH,cases,fresh_cases,round_
from benchmark.expanded_baseline import WORKTREES
from engine.surface_perimeter import _native_alpha


def read_arm(path):
    score=read(path/"score.json")
    coords=read(path/"placements.json")
    assert dict((tuple(p),m) for p,m in coords["local"]) == decoded_pack(path/"reconstruction.zip")
    fixed=score["score"]["fixed"] or score["score"]["relaxed_yaw"]
    return fixed,dict((tuple(p),m) for p,m in coords["aligned"])


def summarize(cases_, roots, prefix):
    totals={}
    rows=[]
    for case in cases_:
        frame=Path(case["frame"])
        truth={tuple(f["xyz_world"]) for f in read(frame/"labels.json")["faces"]}
        results={}
        delta={}
        for i,(arm,root) in enumerate(roots.items()):
            score,blocks=read_arm(root/prefix/case["id"])
            results[arm]=score
            if i==0:
                previous=blocks.keys()
            delta[arm]=dict(gained=len((blocks.keys()-previous)&truth),lost=len((previous-blocks.keys())&truth),
                added=len((blocks.keys()-previous)-truth),removed=len((previous-blocks.keys())-truth))
            for key in ("all",case["dataset"],case["split"]):
                group=totals.setdefault(key,{}).setdefault(arm,{})
                for field in ("geometry","scores"):
                    for kind in ("strict","facade"):
                        target=group.setdefault(field+"_"+kind,Counter())
                        target.update({k:score[field][kind][k] for k in ("correct","missed","extras")})
                group.setdefault("paired",Counter()).update(delta[arm])
        rows.append(dict(id=case["id"],dataset=case["dataset"],score=results,paired=delta))
    for arms in totals.values():
        for stats in arms.values():
            for k,v in stats.items():
                if k!="paired":
                    v["f1"]=2*v["correct"]/(2*v["correct"]+v["missed"]+v["extras"])
    return dict(rows=rows,cohorts=totals)


def compare_sheet(rows, out):
    font=ImageFont.truetype("arial.ttf",16)
    sheet=Image.new("RGB",(1080,len(rows)*223+25),"#142031")
    draw=ImageDraw.Draw(sheet)
    for i,label in enumerate(("Source image","Whole-wall baseline","Visible horizontal cells")):
        draw.text((360*i+5,4),label,font=font,fill="white")
    for idx,(case,base) in enumerate(rows):
        frame=Path(case["frame"])
        labels=read(frame/"labels.json")["faces"]
        with np.load(frame/"labels.npz") as arr:
            owners=arr["face_id"]
        rgb=Image.open(frame/"rgb.png").convert("RGB")
        for i,root in enumerate((None,base,OUT.parent/"horizontal-cells/assembly")):
            pic=rgb.copy()
            caption=case["id"].split("-",1)[1]
            if root:
                score,placed=read_arm(root/"connected-atlas/sky"/case["id"])
                ids=[f["id"] for f in labels if tuple(f["xyz_world"]) in placed]
                mask=Image.fromarray(np.isin(owners,ids).astype("uint8")*180)
                pic.paste(Image.new("RGB",pic.size,"#2aff91"),mask=mask)
                hits=score["geometry"]["strict"]
                caption=f'{hits["correct"]} correct, {hits["extras"]} extra'
            pic.thumbnail((354,187),Image.Resampling.LANCZOS)
            x,y=i*360+3,28+idx*223
            sheet.paste(pic,(x,y))
            draw.text((x,y+pic.height+2),caption,font=font,fill="white")
    sheet.save(out,quality=90)


if __name__=="__main__":
    review=OUT.parents[1]/"benchmark/reviews/floating-horizontal-bridges-01"
    all_results={}
    samples=[]
    for cohort,items,base in (("previous",list(cases()),OLD),("fresh",list(fresh_cases()),FRESH),
                              ("supplemental",list(fresh_cases(True)),FRESH)):
        if not all((OUT.parent/"horizontal-cells/audit"/(c["id"]+".json")).exists() for c in items):
            continue
        roots={"whole-wall":base/"source_material/assembly","rim-joins":OUT/"assembly",
               "horizontal-cells":OUT.parent/"horizontal-cells/assembly"}
        result=summarize(items,roots,"connected-atlas/sky")
        all_results[cohort]=result
        ranked=sorted(zip(items,result["rows"]),key=lambda p:sum(p[1]["paired"]["horizontal-cells"].values()),reverse=True)
        samples.extend((case,roots["whole-wall"]) for case,_ in ranked[:1])
    sky=read(round_.SHARED/"fixtures.json")["cases"]
    if all((OUT.parent/"six-sky/whole-wall/connected-atlas/sky"/c["id"]/"score.json").exists() for c in sky):
        all_results["sky"]=summarize(sky,{a:OUT.parent/"six-sky"/a for a in ("ordinary","whole-wall","horizontal-cells")},"connected-atlas/sky")
        case=next(c for c in sky if c["id"]=="validation-hillside-chapel-1-warm-low-left")
        handoff=WORKTREES/"floating-visible-builds/runs/floating-cloud-gate-01/sky"/case["id"]/"source-handoff"
        plane=next(p for p in read(handoff/"surfaces.json")["planes"] if p["id"]=="mask-0")
        with Image.open(case["variants"]["sky"]["image"]) as image:
            alpha,(x,y,x1,y1)=_native_alpha(plane["support"],image.size)
        frame=Path(case["frame"])
        faces={f["id"]:f for f in read(frame/"labels.json")["faces"]}
        with np.load(frame/"labels.npz") as arrays:
            owners=arrays["face_id"][y:y1,x:x1][np.asarray(alpha)>0]
        ids,counts=np.unique(owners,return_counts=True)
        surfaces=Counter()
        for fid,count in zip(ids,counts):
            if fid:
                surfaces[faces[fid]["surface_id"]]+=int(count)
        save(review/"chapel-mask-check.json",dict(id=case["id"],mask=plane["id"],pixels=len(owners),
             background=int((owners==0).sum()),surface_pixels=surfaces,
             visible_blocks=len({tuple(faces[f]["xyz_world"]) for f in ids if f})))
    save(review/"results.json.gz",all_results)
    if samples:
        compare_sheet(samples,review/"placement.jpg")
    lines=["# Placement and material scores", "",
           "Each row pools decoded blocks across its views. F1 is reported as a percentage. "
           "The last column counts correct positions, visible misses and extra placements.", "",
           "| Cohort | Dataset | Method | Position | Facade position | Position + ID | Facade position + ID | Correct / missed / extra |",
           "|---|---|---|---:|---:|---:|---:|---|"]
    for key,res in all_results.items():
        for ds,group in res["cohorts"].items():
            print(key,ds,{m:[round(100*s[k]["f1"],2) for k in ("geometry_strict","geometry_facade","scores_strict","scores_facade")] for m,s in group.items()})
            for method,stats in group.items():
                values=[f'{100*stats[k]["f1"]:.2f}' for k in ("geometry_strict","geometry_facade","scores_strict","scores_facade")]
                g=stats["geometry_strict"]
                counts=" / ".join(f"{g[k]:,}" for k in ("correct","missed","extras"))
                lines.append("| "+" | ".join([key,ds,method,*values,counts])+" |")
    (review/"scores.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
