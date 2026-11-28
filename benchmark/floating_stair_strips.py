"""Detect repeated visible stair edges and isolate their effect on the parent wall chart."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from PIL import Image
from engine.geometry import project
from engine.surface_perimeter import _native_alpha
from benchmark.relative_oracle_charts import fit_homography
from benchmark.floating_one_face_control import extend
from benchmark.floating_line_shear import CASES, OUT, ISOLATION, paths, read, save, combined, LOCATION, summarize

RUN=Path("runs/stair-strips")


def repeated_edges(rgb,mask):
    yy,xx=np.nonzero(mask)
    lo=np.array([xx.min(),yy.min()])
    crop=rgb[lo[1]:yy.max()+1,lo[0]:xx.max()+1]
    gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),(5,5),1.2)
    raw=cv2.HoughLinesP(cv2.Canny(gray,12,34),1,np.pi/180,18,minLineLength=28,maxLineGap=9)
    lines=[]
    for segment in (() if raw is None else raw):
        line=segment.reshape(2,2)+lo
        line=line[np.argsort(line[:,0])]
        dx,dy=line[1]-line[0]
        if dx<18 or abs(dy)>.22*dx:
            continue
        points=np.rint(np.linspace(*line,9)).astype(int)
        if mask[points[:,1],points[:,0]].mean()<.65:
            continue
        lines.append(line)
    lines.sort(key=lambda l:l[:,1].mean())
    # A single material boundary often yields several collinear fragments.
    merged=[]
    for line in lines:
        inserted=False
        for j,prior in enumerate(merged):
            dx,dy=prior[1]-prior[0]
            predicted=prior[0,1]+(line[:,0]-prior[0,0])*dy/dx
            gap=max(line[0,0]-prior[1,0],prior[0,0]-line[1,0],0)
            if np.max(abs(line[:,1]-predicted))<=2.3 and gap<=13:
                points=np.concatenate([prior,line])
                merged[j]=points[np.argsort(points[:,0])[[0,-1]]]
                inserted=True
                break
        if not inserted:
            merged.append(line)
    longest=sorted(merged,key=lambda l:np.ptp(l[:,0]),reverse=True)[:48]
    lines=sorted(longest,key=lambda l:l[:,1].mean())
    arr=np.asarray(lines)
    centers=np.array([l[:,1].mean() for l in lines])
    # Fit a regular sequence of line centers, allowing one edge to be interrupted.
    # Verify each intermediate edge independently. Missing edges supply no new pixels.
    chain=[]
    for i,first in enumerate(lines):
        for j,last in enumerate(lines[i+4:],start=i+4):
            w=np.ptp(first[:,0]); drift=last[:,0]-first[:,0]
            if drift.prod()<=0 or min(abs(drift))<.25*w or not .6<np.ptp(last[:,0])/w<1.8:
                continue
            for steps in range(4,min(14,j-i)+1):
                dy=(last[:,1].mean()-first[:,1].mean())/steps
                if not .09*w<dy<.36*w:
                    continue
                hits=[i]
                for k in range(1,steps):
                    t=k/steps
                    expected=first*(1-t)+last*t
                    subset=slice(hits[-1]+1,j)
                    if subset.start>=j:
                        break
                    cy=abs(centers[subset]-expected[:,1].mean())/dy
                    error=np.abs(arr[subset,:,0]-expected[:,0])/w
                    valid=(cy<=.34)&(error.min(1)<.22)&(error.max(1)<.55)
                    if not valid.any():
                        break
                    hits.append(subset.start+int(np.argmin(np.where(valid,cy+error.mean(1),np.inf))))
                hits.append(j)
                if len(hits)==steps+1 and len(hits)>len(chain):
                    chain=hits
    if len(chain)<5:
        return []
    return [lines[i] for i in chain]


def run():
    output={}
    for panel,name,wall,_ in CASES:
        src,_,_,cases=paths(panel)[0]
        case=next(c for c in cases if c["id"]==name)
        rel=Path("sky")/name/"charts.json"
        base=ISOLATION/panel/"large-contraction"
        raw=read(base/"charts"/rel)
        old=next(c for c in raw["charts"] if c["id"]==wall)
        with Image.open(case["variants"]["sky"]["image"]) as im:
            rgb=np.asarray(im.convert("RGB"))
        pl=next(p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
        alpha,(x,y,r,b)=_native_alpha(pl["support"],rgb.shape[1::-1])
        mask=np.zeros(rgb.shape[:2],bool)
        mask[y:b,x:r]=np.asarray(alpha)>0
        edges=repeated_edges(rgb,mask)
        charts=[old]
        accepted=[]
        segmented=np.zeros_like(mask)
        for i,(top,bottom) in enumerate(zip(edges,edges[1:])):
            # Fit each visible band in its local source corners; retain the parent wall's
            # basis convention so the ordinary joining procedure can evaluate these charts.
            quad=np.array([bottom[0],bottom[1],top[1],top[0]])
            uv=project(np.linalg.inv(old["H"]),quad)
            n=int(np.rint(np.linalg.norm(uv[1]-uv[0])))
            if not 1<=n<=32:
                continue
            h=fit_homography([0,0],quad)@np.diag([1/n,1.,1.])
            piece=np.zeros(mask.shape,"uint8")
            cv2.fillConvexPoly(piece,np.rint(quad).astype("int32"),1)
            support=piece.astype(bool)&mask
            child=extend(old,h,support,rgb)
            if not child["cells"]:
                continue
            child.update(id=f"{wall}-step-{i}",confidence=old["confidence"])
            charts.append(child)
            segmented|=support
            accepted.append(dict(chart=child["id"],quad=quad.tolist(),cells=len(child["cells"])))
        if accepted:
            points=np.array([[c["u"]+.5,c["v"]+.5] for c in old["cells"]])
            centers=np.floor(project(np.asarray(old["H"]),points)).astype(int)
            inside=(centers[:,0]>=0)&(centers[:,0]<mask.shape[1])&(centers[:,1]>=0)&(centers[:,1]<mask.shape[0])
            moved=np.zeros(len(centers),bool)
            moved[inside]=segmented[centers[inside,1],centers[inside,0]]
            charts[0]=dict(old,cells=[c for c,cut in zip(old["cells"],moved,strict=True) if not cut])
        dest=RUN/panel
        packet=dict(raw,charts=[item for c in raw["charts"] for item in (charts if c["id"]==wall else [c])])
        save(dest/"charts"/rel,packet)
        # Child IDs do not have a separate predicted normal field. They inherit the
        # parent's direction and can independently remain unjoined by the assembler.
        with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
            combined.run((panel,case,dest/"charts",dest/"assembly"))
        score=summarize([case],{"before":base/"assembly","after":dest/"assembly"},str(LOCATION))["rows"][0]
        output[name]=dict(wall=wall,edges=len(edges),children=accepted,score=score)
        print(name,len(accepted),score["paired"]["after"],flush=True)
    save(OUT/"stair-strips.json.gz",output)


if __name__=="__main__":
    run()
