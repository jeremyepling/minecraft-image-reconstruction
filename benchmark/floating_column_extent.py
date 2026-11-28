"""Recheck visible repeated base extents and anchor repeated post fronts at the sky edge."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch as mock_patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_column_recovery import BASE, EXTENDED, LOCATION, OUT, PANELS, REVIEW, base_root, paths
from benchmark.floating_depth_refit import SAMPLES, TEXELS, atlas, combined, project, rank_tile, read, sample, save, summarize, totals
from engine.surface_perimeter import _native_alpha

ARMS=("base-extent","silhouette-on-parent-depth","silhouette-on-band-depth","both-extents")


def crop_grid(h,low,high,rate=16):
    u=np.arange(low[0],high[0],1/rate)
    v=np.arange(low[1],high[1],1/rate)
    grid=np.stack(np.meshgrid(u,v),-1)
    return u,v,grid,project(h,grid)


def pixels(image,xy):
    ix,iy=np.floor(xy).astype(int).transpose(2,0,1)
    return image[iy.clip(0,len(image)-1),ix.clip(0,image.shape[1]-1)]


def tops(patch,h,native,rgb):
    uv=np.array(patch["old_uv"])
    lo,hi=uv.min(0),uv.max(0)
    if patch["depth"]!=1 or hi[0]-lo[0]<6 or not 1<=hi[1]-lo[1]<=4:
        return []
    shoulder=hi[1]+1
    u,v,_,xy=crop_grid(h,[lo[0]-1,shoulder+.03],[hi[0]+2,shoulder+4.1])
    colors=pixels(rgb,xy).astype(int)
    blue=(colors[...,2]>192)&(colors[...,2]-colors[...,0]>38)&(colors[...,2]-colors[...,1]>13)
    supported=sample(native,xy,1)&~blue
    upper=np.where(supported,v[:,None],0).max(0)
    sky_above=np.argmax((v[:,None]>upper+.06),axis=0)
    reaches=(upper>=shoulder+1.4)&(upper<shoulder+3.7)&blue[sky_above,np.arange(len(u))]
    ix=np.flatnonzero(reaches)
    if not len(ix):
        return []
    peaks=[]
    for indices in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
        w=len(indices)/16
        if not .55<=w<=1.85:
            continue
        # Only use the horizontal crest; an overlapping rear post can extend its lower edge.
        top=np.quantile(upper[indices],.8)
        crest=indices[upper[indices]>=top-.18]
        if not len(crest) or (crest[-1]-crest[0])/16>1.5:
            continue
        peaks.append([np.median(u[crest]),top,(crest[-1]-crest[0]+1)/16])
    if len(peaks)<3:
        return []
    centers=np.array(peaks)[:,0]
    period=round(np.median(np.diff(centers)))
    if not 2<=period<=6:
        return []
    rank=np.rint((centers-centers[0])/period).astype(int)
    if len(set(rank))<len(rank) or np.max(abs(centers-centers[0]-rank*period))>.95:
        return []
    result=[]
    for (middle,top,width),i in zip(peaks,rank):
        base_u=int(np.floor(centers[0])+period*i)
        n=int(round(top-shoulder))
        if not 1<=n<=4:
            continue
        # The observed silhouette defines the local upper edge and local face width.
        width=np.clip(width,.75,1.25)
        edge=middle-width/2
        transform=np.array([[width,0,edge-width*base_u],[0,1,top-(shoulder+n)],[0,0,1.]])
        child=h@transform
        cells=[]
        for v0 in range(shoulder,shoulder+n):
            where=project(child,[base_u,v0]+SAMPLES)
            if sample(native,where,1).mean()>.5:
                cells.append([int(base_u),int(v0)])
        if cells:
            result.append(dict(H=child.tolist(),uv=cells,step=1,center=float(middle),sky_v=float(top),width=float(width)))
    return result


def bases(patches,h,native,rgb):
    if len(patches)<2:
        return []
    parts=[]
    for part in patches:
        old=np.array(part["old_uv"])
        lo,hi=old.min(0),old.max(0)
        if (hi-lo).max()>3 or (hi-lo).min()>2:
            continue
        u,v,g,xy=crop_grid(h,lo-1.25,hi+2.25)
        valid=sample(native,xy,1)
        cols=pixels(rgb,xy).astype(float)
        keys=np.floor(g).astype(int)
        in_part=np.zeros(valid.shape,bool)
        for key in old:
            in_part|=(keys==key).all(2)
        vals=cols[in_part & valid]
        if len(vals)<20:
            continue
        med=np.median(vals,0)
        # Remove luminance from one component of the metric to tolerate shading on matching grain.
        delta=cols-med
        compatible=valid & (np.linalg.norm(delta-delta.mean(2,keepdims=True),axis=2)<9) & (abs(delta.mean(2))<24)
        _,labels=cv2.connectedComponents(compatible.astype("uint8"),connectivity=8)
        regions,counts=np.unique(labels[in_part & valid],return_counts=True)
        use=regions!=0
        if not use.any():
            continue
        region=labels==regions[use][np.argmax(counts[use])]
        if np.count_nonzero(region & in_part)<.55*np.count_nonzero(valid & in_part):
            continue
        pts=g[region]
        a,b=np.quantile(pts,[.02,.98],axis=0)
        parts.append(dict(old=old,lo=a,hi=b,median=med,step=part["depth"],region=region,u=u,v=v))
    result=[]
    for part in parts:
        peers=[q for q in parts if q is not part and q["step"]==part["step"] and
               np.linalg.norm(q["median"]-part["median"])<=20 and abs(q["hi"][1]-part["hi"][1])<=.8]
        if not peers:
            continue
        a,b=part["lo"],part["hi"]
        width=int(round(float(np.median([q["hi"][0]-q["lo"][0] for q in [*peers,part]]))))
        if not 1<=width<=3 or abs(b[0]-a[0]-width)>.6:
            continue
        # Choose the whole-block horizontal origin nearest the old patch with maximum shared cells.
        old=part["old"]; left=int(round(a[0]))
        options=range(left-1,left+2)
        left=min(options,key=lambda n:(-sum(n<=x<n+width for x,_ in old),abs(n-a[0])))
        local_width=(b[0]-a[0])/width
        child=h@np.array([[local_width,0,a[0]-local_width*left],[0,1,0],[0,0,1]])
        cells=[]
        for j in sorted(set(old[:,1].tolist())):
            for i in range(left,left+width):
                positions=project(np.linalg.solve(h,child),[i,j]+SAMPLES)
                ix=np.rint((positions[...,0]-part["u"][0])*16).astype(int)
                iy=np.rint((positions[...,1]-part["v"][0])*16).astype(int)
                inside=(ix>=0)&(iy>=0)&(ix<len(part["u"]))&(iy<len(part["v"]))
                ok=inside & part["region"][iy.clip(0,len(part["v"])-1),ix.clip(0,len(part["u"])-1)]
                if ok.mean()>.5:
                    cells.append([i,j])
        if cells:
            result.append(dict(H=child.tolist(),uv=cells,step=part["step"],old=old.tolist()))
    return result


def run(job):
    panel,case,previous,depth=job
    name=case["id"]; rel=Path("sky")/name/"charts.json"
    root=OUT/panel/"repeated-texture-attachment"
    if not (root/"charts"/rel).exists():
        root=base_root(panel,name)
    packet=read(root/"charts"/rel)
    regions=defaultdict(lambda:dict(tops=[],bases=[]))
    materials={}
    if depth["patches"]:
        im=Image.open(case["variants"]["sky"]["image"]).convert("RGB"); rgb=np.asarray(im)
        source=paths(panel)[0][0]
        planes={p["id"]:p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"]}
        for ch in packet["charts"]:
            cid=ch["id"]
            parts=[p for p in depth["patches"] if p["chart"]==cid]
            if not parts: continue
            h=np.array(ch["H"])
            at=project(h,np.array([[0,0],[1,0],[0,1]]))
            if at[1,0]<=at[0,0] or at[2,1]>=at[0,1]: continue
            alpha,(x,y,r,b)=_native_alpha(planes[cid]["support"],im.size)
            native=np.zeros(rgb.shape[:2],bool);native[y:b,x:r]=np.asarray(alpha)>0
            regions[cid]["tops"]=[t for p in parts for t in tops(p,h,native,rgb)]
            small=[p for p in depth["extra_patches"] if p["chart"]==cid]
            regions[cid]["bases"]=bases(small,h,native,rgb)
            for fitted in regions[cid]["tops"]+regions[cid]["bases"]:
                for u,v in fitted["uv"]:
                    xy=project(np.array(fitted["H"]),TEXELS+[u,v])
                    hit=rank_tile(dict(rgb=pixels(rgb,xy),mask=sample(native,xy,1)),atlas())
                    materials[id(fitted),u,v]=dict(block_id=hit["block_id"] or "smooth_stone",fallback=not hit["block_id"])
    roots={"attachment":root/"assembly"}
    for arm in ARMS:
        changed=False
        charts=[]
        for ch in packet["charts"]:
            scope=regions[ch["id"]]
            top=scope["tops"] if arm!="base-extent" else []
            bottom=scope["bases"] if arm in ("base-extent","both-extents") else []
            cells=ch.get("export_cells",ch["cells"])
            if not top and not bottom:
                charts.append(ch);continue
            updated={(c["u"],c["v"],c.get("normal_offset",0)):c for c in cells}
            for patch in bottom+top:
                for u,v in patch.get("old",[]):
                    updated.pop((u,v,patch["step"]),None)
                for u,v in patch["uv"]:
                    if patch in top:
                        updated.pop((u,v,0),None)
                    offset=0 if arm=="silhouette-on-parent-depth" else patch["step"]
                    cell=materials[id(patch),u,v]
                    updated[u,v,offset]=dict(cell,u=u,v=v,normal_offset=offset)
            charts.append(dict(ch,export_cells=list(updated.values())));changed=True
        if not changed:
            roots[arm]=roots["attachment"];continue
        target=OUT/panel/arm
        save(target/"charts"/rel,dict(packet,charts=charts))
        with mock_patch.object(combined,"inputs",lambda _:paths(panel)[0]):
            combined.run((panel,case,target/"charts",target/"assembly"))
        roots[arm]=target/"assembly"
    row=summarize([case],roots,str(LOCATION))["rows"][0]
    assert row["score"]["attachment"]==previous["score"]["repeated-texture-attachment"]
    row.update(panel=panel,layout=previous["layout"],active=previous["active"],source_sha256=previous["source_sha256"],
               fitted={k:v for k,v in regions.items() if v["tops"] or v["bases"]})
    save(OUT/panel/"extent-cases"/(name+".json.gz"),row)


def main():
    depth={(r["panel"],r["id"]):r for r in read(BASE/"benchmark/reviews/floating-depth-refit-01/conditions.json.gz")}
    baseline={(r["panel"],r["id"]):r for r in read(REVIEW/"conditions.json.gz")}
    jobs=[(p,c,baseline[p,c["id"]],depth[p,c["id"]]) for p in PANELS for c in paths(p)[0][3]]
    with ProcessPoolExecutor(6) as pool: list(pool.map(run,jobs))
    rows=[read(f) for p in PANELS for f in sorted((OUT/p/"extent-cases").glob("*.json.gz"))]
    assert len(rows)==92
    save(REVIEW/"extent-cases.json.gz",rows)
    save(REVIEW/"extent-summary.json",{k:totals(g) for k,g in dict(
        active=[r for r in rows if r["active"]],sky=[r for r in rows if r["panel"]=="sky"]).items()})


if __name__=="__main__":
    main()
