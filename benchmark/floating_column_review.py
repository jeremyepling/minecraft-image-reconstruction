"""Show Viaduct's detected column, discarded chart and recovered decoded coordinates."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_column_recovery import LOCATION, OUT, REVIEW, UNIT, base_root, paths, project, read, save
from benchmark.floating_bridges_report import read_arm
from engine.surface_perimeter import _native_alpha


def main():
    from pathlib import Path
    panel,name="holdout4","train-viaduct-01-riverstone-low-left"
    source,ordinary,_,cases=paths(panel)[0]
    case=next(c for c in cases if c["id"]==name)
    base=base_root(panel,name)
    row=read(OUT/panel/"cases"/(name+".json.gz"))
    rel=Path("sky")/name/"charts.json"
    packet=read(base/"charts"/rel)
    chart=next(c for c in packet["charts"] if c["id"]=="mask-1")
    native=read(Path(case["frame"])/"labels.json")
    with np.load(Path(case["frame"])/"labels.npz") as labels:
        owners=labels["face_id"]
    im=Image.open(case["variants"]["sky"]["image"]).convert("RGB")
    planes={p["id"]:p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    masks={}
    for cid in ("mask-0","mask-1"):
        alpha,(x,y,r,b)=_native_alpha(planes[cid]["support"],im.size)
        mask=np.zeros(owners.shape,bool)
        mask[y:b,x:r]=np.asarray(alpha)>0
        masks[cid]=mask
    h=np.asarray(chart["H"])
    uv=np.array([(c["u"],c["v"]) for c in chart["cells"]])
    centers=project(h,uv+.5).astype(int)
    ids=owners[centers[:,1],centers[:,0]]
    seen={c["id"]:c for c in native["faces"]}
    truth=np.isin(owners,ids)
    _,before=read_arm(base/"assembly"/LOCATION/name)
    _,after=read_arm(OUT/panel/"repeated-texture-attachment/assembly"/LOCATION/name)
    audit=dict(chart=chart["id"],source_grid_H=h.tolist(),uv=uv.tolist(),world=[seen[i]["xyz_world"] for i in ids],
               matched_faces=len(set(ids.tolist())),source_mask_pixels=int((truth&masks["mask-1"]).sum()),
               parent_mask_pixels=int((truth&masks["mask-0"]).sum()),native_pixels=int(truth.sum()),
               prior_correct=sum(tuple(seen[i]["xyz_world"]) in before for i in ids),
               candidate_correct=sum(tuple(seen[i]["xyz_world"]) in after for i in ids))
    save(REVIEW/"viaduct-trace.json",audit)
    xy=project(h,uv[:,None,:]+UNIT).reshape(-1,2)
    target=tuple(np.r_[xy.min(0)-14,xy.max(0)+14].astype(int))
    views=[im.copy() for _ in range(4)]
    draw=ImageDraw.Draw(views[0])
    draw.rectangle(target,outline="#43e8d1",width=5)
    mask=Image.fromarray((masks["mask-1"]*130).astype("uint8"))
    views[1].paste(Image.new("RGB",im.size,"#5ae5d3"),mask=mask)
    for i,points in enumerate(project(h,uv[:,None,:]+UNIT)):
        pts=list(map(tuple,points))
        for v,color in ((2,"#ffc753"),(3,"#4ff9b3" if tuple(seen[ids[i]]["xyz_world"]) in after else "#ef7180")):
            ImageDraw.Draw(views[v]).line(pts+pts[:1],fill=color,width=2)
    titles=("Source image","Surface Finder · mask-1","Counted chart · discarded","Attached in decoded export")
    captions=("Far-left column beneath the band","48,486 of 48,791 visible face pixels",
              "24 counted front faces, 0 correct before","24 recovered, including 4 log fronts")
    font=ImageFont.truetype("arial.ttf",18)
    small=ImageFont.truetype("arial.ttf",16)
    sheet=Image.new("RGB",(1400,490),"#17212b")
    d=ImageDraw.Draw(sheet)
    for i,view in enumerate(views):
        if i: view=view.crop(target)
        else: view=view.crop((195,75,1150,755))
        view.thumbnail((342,400),Image.Resampling.LANCZOS)
        sheet.paste(view,(350*i+(350-view.width)//2,39+(400-view.height)//2))
        d.text((350*i+8,9),titles[i],font=font,fill="white")
        d.text((350*i+8,455),captions[i],font=small,fill="white")
    sheet.save(REVIEW/"viaduct-column-recovery.jpg",quality=92)
    ext=read(OUT/panel/"extent-cases"/(name+".json.gz"))
    fitted=ext["fitted"]["mask-0"]
    compare=Image.new("RGB",(1360,552),"#17212b")
    draw=ImageDraw.Draw(compare)
    for i,title in enumerate(("Before · attached column","After · visible base edges and post tops")):
        picture=im.copy()
        pen=ImageDraw.Draw(picture)
        def lines(matrix,cells,color):
            for points in project(np.asarray(matrix),np.asarray(cells)[:,None,:]+UNIT):
                pts=list(map(tuple,points))
                pen.line(pts+pts[:1],fill=color,width=3)
        lines(audit["source_grid_H"],audit["uv"],"#76b9ff")
        if i:
            for part in fitted["tops"]:
                lines(part["H"],part["uv"],"#ffc04b")
            for part in fitted["bases"]:
                lines(part["H"],part["uv"],"#54f5d0")
        else:
            chart=next(c for c in packet["charts"] if c["id"]=="mask-0")
            lines(chart["H"],[(c["u"],c["v"]) for c in chart["cells"] if c["v"]>=7],"#ffc04b")
            for part in fitted["bases"]:
                lines(chart["H"],part["old"],"#54f5d0")
        view=picture.crop((208,72,1140,754))
        view.thumbnail((664,466),Image.Resampling.LANCZOS)
        compare.paste(view,(680*i+(680-view.width)//2,38+(466-view.height)//2))
        draw.text((680*i+12,10),title,font=font,fill="white")
        score=ext["score"]["both-extents" if i else "attachment"]["geometry"]["strict"]
        draw.text((680*i+12,505),f'{score["correct"]} correct     {score["extras"]} extra',font=font,fill="white")
    compare.save(REVIEW/"viaduct-visible-extents.jpg",quality=94)


if __name__=="__main__":
    main()
