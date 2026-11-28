"""Render identical source crops with recorded owner and chart overlays."""
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit import OUT, PIN, STABLE, PARTS, paths, read, surface_mask, archive_item
from engine.geometry import project

items=[
    ("holdout3","train-stable-yard-1-warm-high-right","mask-0~p24","Stable Yard roof step",False),
    ("original","test-castle-02-ashlar-high-right","mask-0","Castle front relief",False),
    ("sky","train-viaduct-02-riverstone-high-right","mask-0","Viaduct wall and ledge",False),
    ("sky","train-pillars-01-marble-low-left","mask-1~p7","Pillars same-depth material changes",True),
    ("sky","test-walled-garden-1-warm-high-right","mask-1-wall-1","Walled Garden whole chart depth shift",False),
]
pin=read(PIN/"manifest.json")
inputs={(r["panel"],r["id"]):r for r in pin["conditions"]}
charts={(r["panel"],r["id"]):{c["id"]:c for c in r["charts"]} for r in read(OUT/"full-audit.json")["views"]}
face_palette=[(60,217,194),(247,191,73),(219,113,190),(143,156,245)]
status_color={"behind_native_plane":(102,168,255),"in_front_of_native_plane":(255,106,150),
              "right_plane_other_coordinate":(255,212,96),"different_normal":(180,105,230),
              "background_or_off_frame":(230,234,237),"right_coordinate_wrong_material":(75,215,189),
              "right_coordinate_exact":(75,215,189)}
ink="#f0f6fb"
muted="#bdcedb"
large=ImageFont.truetype("arialbd.ttf",25)
heading=ImageFont.truetype("arialbd.ttf",18)
small=ImageFont.truetype("arial.ttf",15)
ww,hh=470,234
stride=486
row_h=306
sheet=Image.new("RGB",(1474,103+row_h*len(items)),"#15212e")
draw=ImageDraw.Draw(sheet)
draw.text((15,8),"Which photographed plane did the grid sample?",fill=ink,font=large)
for i,title in enumerate(("Source crop","Native planes and materials","Grid cells colored by assigned depth")):
    draw.text((15+stride*i,43),title,font=heading,fill=muted)
draw.text((15,68),"Each row uses identical crops. Colors in the middle identify recorded pixel owners. Grid colors compare the exported cell with its source owner.",font=small,fill=muted)

unit=np.array(((0,0),(1,0),(1,1),(0,1)))
for i,(p,n,cid,title,by_mat) in enumerate(items):
    row=inputs[p,n]
    c=charts[p,n][cid]
    rel=STABLE/row["reference"]["labels.json"]["path"]
    labels=read(rel)
    truth={tuple(b["xyz_world"]):b["block_id"] for b in labels["blocks"]}
    with np.load(STABLE/row["reference"]["labels.npz"]["path"],allow_pickle=False) as z:
        owners=z["face_id"]
    source=Image.open(STABLE/row["source"]["path"]).convert("RGB")
    src=paths(p)[0][0]/"sky"/n/"source-handoff/surfaces.json"
    source_id=c["source_id"]
    plane=next(r for r in read(src)["planes"] if r["id"]==source_id)
    mask=surface_mask(plane,source.size)
    if c["region"] is not None:
        native=PARTS/(n+"-"+source_id+"-ids.png")
        part=np.asarray(Image.open(native))
        mask=mask & (part==c["region"])
    painted=np.array(source)
    visible=mask & (owners!=0)
    yy,xx=np.nonzero(visible)
    x0=max(0,int(xx.min())-35);y0=max(0,int(yy.min())-35)
    x1=min(source.width,int(xx.max())+36);y1=min(source.height,int(yy.max())+36)
    # Supply the same crop for all columns and retain margins around the scored mask.
    if x1-x0<280:
        ext=(280-(x1-x0))//2
        x0=max(0,x0-ext);x1=min(source.width,x1+ext)
    if y1-y0<150:
        ext=(150-(y1-y0))//2
        y0=max(0,y0-ext);y1=min(source.height,y1+ext)
    crop=(x0,y0,x1,y1)
    defs={f["id"]:((f["direction"],round(float(np.dot(f["corners_world"][0],f["normal_world"])),5)),
          truth[tuple(f["xyz_world"])].removeprefix("minecraft:")) for f in labels["faces"]}
    if by_mat:
        physical=tuple(c["source_mask"]["dominant_same_direction"])
        members=sorted(c["source_mask"]["same_plane_material_pixels"].items(),key=lambda p:-p[1])[:3]
        lookup=[(physical,k.removeprefix("minecraft:"),f"{k.removeprefix('minecraft:').replace('_',' ')}") for k,_ in members]
    else:
        lookup=[((d["direction"],d["signed_depth"]),None,f'{d["direction"]} {d["signed_depth"]:g}') for d in c["source_mask"]["major_planes"][:4]]
    for (physical,mat,name),rgb in zip(lookup,face_palette):
        selected=mask & np.isin(owners,[id for id,(v,m) in defs.items() if v==physical and (mat is None or m==mat)])
        painted[selected]=painted[selected]*.36+np.array(rgb)*.64
    pix=source.copy()
    dd=ImageDraw.Draw(pix,"RGBA")
    with ZipFile(PIN/"exports.zip") as z:
        cp=json.loads(archive_item(z,row,"charts"))
    h=np.array(next(r["H"] for r in cp["charts"] if r["id"]==cid))
    sites=[s for s in read(OUT/"conditions"/p/(n+"-sites.json.gz")) if s["chart"]==cid]
    for s in sites:
        polygon=[tuple(hh) for hh in project(h,unit+s["uv"])]
        color=status_color[s["result"]]
        dd.line(polygon+[polygon[0]],fill=(*color,175),width=1)
        x,y=s["xy"]
        if s["extra"] or s["result"] in ("behind_native_plane","in_front_of_native_plane","different_normal"):
            dd.ellipse((x-1.45,y-1.45,x+1.45,y+1.45),fill=(*color,255))
    y=102+i*row_h
    draw.text((15,y),title,font=heading,fill=ink)
    y+=25
    for j,img in enumerate((source,Image.fromarray(painted),pix)):
        ch=img.crop(crop)
        ch.thumbnail((ww,hh),Image.Resampling.LANCZOS)
        sheet.paste(ch,(15+stride*j,y))
    x=15+stride
    for (_,_,name),v in zip(lookup,face_palette):
        draw.rectangle((x,y+hh+8,x+9,y+hh+17),fill=v)
        draw.text((x+14,y+hh+3),name,font=small,fill=muted)
        x+=int(draw.textlength(name,font=small))+26
    numbers=c["center_vs_placement"]
    if by_mat:
        draw.text((15+stride,y+hh+26),"Different materials on the same south-facing plane at z=2.",font=small,fill=muted)
    else:
        draw.text((15+stride,y+hh+26),f'{c["source_mask"]["visible_pixels"]:,} native face pixels in this mask or split part.',font=small,fill=muted)
    draw.text((15,y+hh+5),f'{p} • {cid} • {c["accepted_exported"]:,} emitted blocks',font=small,fill=muted)
    depth=numbers.get("behind_native_plane",0)+numbers.get("in_front_of_native_plane",0)
    draw.text((15+2*stride,y+hh+6),f'{depth:,} on a different aligned depth from the source owner.',fill=muted,font=small)
    for j,(lab,col) in enumerate((("Behind","behind_native_plane"),("In front","in_front_of_native_plane"),
                                     ("Other facing","different_normal"),("Same depth / wrong site","right_plane_other_coordinate"))):
        lx=15+2*stride+[0,75,158,260][j]
        draw.rectangle((lx,y+hh+29,lx+9,y+hh+38),fill=status_color[col])
        draw.text((lx+13,y+hh+24),lab,font=small,fill=muted)
sheet.save(OUT/"native-plane-examples.png",optimize=True)
