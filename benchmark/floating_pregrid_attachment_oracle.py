"""Reference-only rigid placement ceiling for fresh grids detached from the adopted build."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark.floating_pregrid_depth_refit import OUT,PREFIX,REVIEW,combined,data,project,read,save,summarize,workload
from benchmark.relative_face_assembly import axis_vector,oriented
from benchmark.relative_face_scoring import _yaw


def run(job):
    g,c,_,reviewed,transfer=job
    n=c["id"]
    r=read(OUT/g/"rows"/(n+".json.gz"))
    if not r["audit"] or not (reviewed or transfer):
        return
    base=Path(r["roots"]["noop"])
    adopted=read(OUT/g/"noop/charts/sky"/n/"charts.json")
    fixed=r["score"]["noop"]
    quarter,t=fixed["yaw_quarters"],np.array(fixed["translation"])
    labels=read(Path(c["frame"])/"labels.json")
    faces={f["id"]:f for f in labels["faces"]}
    with np.load(Path(c["frame"])/"labels.npz") as a:
        native=a["face_id"]
    sampled={}
    for cid,record in r["audit"].items():
        if cid=="decision":continue
        for k,part in record["parts"].items():
            for arm in ("complete","partial"):
                if arm in part["grids"]:
                    sampled[cid+"~p"+str(k)]=arm
                    break
    inserts=[]
    proof=[]
    for cid,arm in sampled.items():
        packet=read(OUT/g/arm/"charts/sky"/n/"charts.json")
        child=next(v for v in packet["charts"] if v["id"]==cid)
        trial=dict(child,id="native-shift:"+cid)
        # Reorient relative to the adopted camera-normal anchor. Reject any
        # separate fit that would itself change that orientation.
        oriented_base,hint=oriented(adopted["charts"])
        with_child,after=oriented([*adopted["charts"],trial])
        if hint!=after:continue
        child=next((v for v in with_child if v["id"]==trial["id"]),None)
        if child is None:continue
        allowed=set(map(tuple,child["export_uv"])) if "export_uv" in child else None
        cells=[v for v in child["cells"] if allowed is None or (v["u"],v["v"]) in allowed]
        uu,vv=map(axis_vector,child["axes"]);normal=axis_vector(child["normal_axis"])
        offset=np.minimum(0,uu)+np.minimum(0,vv)-np.maximum(0,normal)
        sites=np.array([(v["u"],v["v"]) for v in cells])
        if not len(sites):continue
        center=np.floor(project(np.asarray(child["H"]),sites+.5)).astype(int)
        votes=[]
        for cell,(x,y) in zip(cells,center):
            if not(0<=y<native.shape[0] and 0<=x<native.shape[1]):continue
            f=faces.get(int(native[y,x]))
            if not f or f["normal_world"]!=list(_yaw(normal,quarter)):continue
            local=np.array(_yaw(np.asarray(f["xyz_world"])-t,(-quarter)%4))
            votes.append(tuple(local-offset-cell["u"]*uu-cell["v"]*vv))
        if not votes:
            proof.append(dict(chart=cid,cells=len(cells),votes=0));continue
        origin,count=Counter(votes).most_common(1)[0]
        accepted=count>=max(2,.66*len(cells))
        proof.append(dict(chart=cid,cells=len(cells),votes=len(votes),agreement=count,
                          origin=list(map(int,origin)),accepted=accepted))
        if accepted:
            for cell in cells:
                pos=tuple((origin+offset+cell["u"]*uu+cell["v"]*vv).tolist())
                inserts.append((pos,cell))
    assemble=combined.assembly.assemble
    def oracle(charts,**kwargs):
        res=assemble(charts,**kwargs)
        if res["groups"]:
            selected=res["groups"][0]
            for pos,cell in inserts:
                if pos not in selected["blocks"]:
                    selected["blocks"][pos]=cell["block_id"]
                    if not cell.get("fallback"):selected["assigned"].add(pos)
        return res
    dst=OUT/g/"native-shift"/"assembly"
    with patch.object(combined.assembly,"assemble",oracle),patch.object(combined,"inputs",data):
        combined.run((g,c,OUT/g/"noop/charts",dst))
    scored=summarize([c],dict(adopted=base,native_shift=dst),PREFIX)["rows"][0]
    result=dict(**scored,panel=g,proposals=proof)
    save(OUT/g/"native"/(n+".json.gz"),result)
    print(n,proof,scored["paired"]["native_shift"],flush=True)
    return result


if __name__=="__main__":
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(run,[j for j in workload() if j[3] or j[4]]))
    save(REVIEW/"reference-attachment.json.gz",[r for r in rows if r])
