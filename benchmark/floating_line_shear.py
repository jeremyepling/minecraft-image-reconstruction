"""Retain block scale while refining a wall grid's shear and origin from native lines."""
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image

from benchmark.floating_extra_budget import paths, combined, LOCATION
from benchmark.floating_grid_signoff import CROSS, ISOLATION, SCALE, get_chart
from benchmark.floating_one_face_control import extend
from benchmark.expanded_baseline import read, save
from benchmark.floating_bridges_report import summarize
from benchmark.full_block_lattice import line_camera
from benchmark.camera_structure_scaffold import outline_lines
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

OUT = Path(__file__).resolve().parent/"reviews/floating-grid-signoff-03"
RUN = Path("runs/floating-line-shear")
CASES = [
    ("sky","train-pillars-01-marble-low-left","mask-1",CROSS),
    ("sky","validation-hillside-chapel-1-warm-low-left","mask-0",ISOLATION),
    ("original","test-twinspire2-earth-high-right","mask-2",SCALE),
    ("holdout4","train-riverside-mill-1-warm-low-left","mask-0",ISOLATION),
    ("holdout4","train-viaduct-01-riverstone-low-left","mask-0",ISOLATION),
    ("original","test-castle-01-ashlar-high-right","mask-0",ISOLATION),
    ("holdout2","train-shrine-02-ochre-low-left","mask-0-wall-0",ISOLATION),
]


def fit(rgb,mask,h):
    y,x=np.nonzero(mask)
    x0,y0=x.min(),y.min()
    gray=cv2.cvtColor(rgb[y0:y.max()+1,x0:x.max()+1],cv2.COLOR_RGB2GRAY)
    lines=cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return h,{}
    uvcenter=project(np.linalg.inv(h),np.array([np.median(x),np.median(y)]))
    affine=np.eye(3)
    audit={}
    for axis in range(2):
        samples=[]
        for raw in lines:
            ends=raw.reshape(2,2)+[x0,y0]
            probes=np.linspace(ends[0],ends[1],9).astype(int)
            if mask[probes[:,1],probes[:,0]].mean()<.75:
                continue
            uv=project(np.linalg.inv(h),ends)
            delta=uv[1]-uv[0]
            if not np.isfinite(uv).all() or np.max(abs(uv-uvcenter))>100:
                continue
            length=abs(delta[1-axis])
            if length<1.25 or abs(delta[axis])>.30*length:
                continue
            samples.append((length,uv))
        samples.sort(key=lambda s:-s[0])
        if len(samples)<8:
            continue
        train=np.array([uv for _,uv in samples[::2]])
        test=np.array([uv for _,uv in samples[1::2]])
        def adjust(slope,segments):
            return segments[:,:,axis]-slope*(segments[:,:,1-axis]-uvcenter[1-axis])
        def cost(z):
            return np.minimum(.35,abs(z-np.rint(z))).mean()
        best=(cost(train[:,:,axis]),0.,0.)
        for slope in np.linspace(-.24,.24,49):
            values=adjust(slope,train)
            for phase in np.linspace(-.5,.5,41):
                loss=cost(values-phase)
                if loss<best[0]:
                    best=(loss,slope,phase)
        _,slope,phase=best
        prior=cost(test[:,:,axis])
        now=cost(adjust(slope,test)-phase)
        agrees=prior-now>.025
        audit[str(axis)]=dict(lines=len(samples),held_out_before=float(prior),held_out_after=float(now),
                               slope=float(slope),phase=float(phase),applied=bool(agrees))
        if agrees:
            affine[axis,1-axis]=-slope
            affine[axis,2]=slope*uvcenter[1-axis]-phase
    return h@np.linalg.inv(affine),audit


def outline_orientation(h, rgb, planes, cells):
    image = Image.fromarray(rgb)
    lines, _ = outline_lines(image,planes,[])
    axes = line_camera(lines,image.size) if len(lines)>=3 else None
    if axes is None or not cells:
        return h
    center=np.median([[c["u"]+.5,c["v"]+.5] for c in cells],axis=0)
    pixel=project(h,center)
    local=(h[:2,:2]-pixel[:,None]*h[2,:2])/(h@[float(center[0]),float(center[1]),1])[2]
    directions=axes[:2]-pixel[:,None]*axes[2]
    norms=np.linalg.norm(directions,axis=0)
    if np.min(norms)<1e-6:
        return h
    match=local.T@(directions/norms)
    # Identify vertical and lateral grid axes jointly. Require both directions to
    # be within 30 degrees of the initial observed wall to avoid swapping facades.
    pairs=[(abs(match[i,1])/np.linalg.norm(local[:,i])+abs(match[1-i,j])/np.linalg.norm(local[:,1-i]),i,j)
           for i in (0,1) for j in (0,2)]
    _,i,j=max(pairs)
    columns=np.zeros((3,3))
    for index,axis in ((i,1),(1-i,j)):
        similarity=abs(match[index,axis])/np.linalg.norm(local[:,index])
        if similarity<np.cos(np.pi/6):
            return h
        columns[:,index]=axes[:,axis]*np.sign(match[index,axis])*np.linalg.norm(local[:,index])/norms[axis]
    columns[:,2]=np.r_[pixel,1]-columns[:,:2]@center
    return columns


def run():
    results={}
    for panel,name,wall,root in CASES:
        (src,_,_,cases),_,_=paths(panel)
        case=next(c for c in cases if c["id"]==name)
        rel=Path("sky")/name/"charts.json"
        a=root/panel
        if root==ISOLATION:
            a=a/"large-contraction"
        raw=read(a/"charts"/rel)
        chart=get_chart(a,rel,wall)
        source=Path(case["variants"]["sky"]["image"])
        with Image.open(source) as im:
            rgb=np.array(im.convert("RGB"))
        part=next(p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
        alpha,(x,y,r,b)=_native_alpha(part["support"],rgb.shape[1::-1])
        mask=np.zeros(rgb.shape[:2],bool)
        mask[y:b,x:r]=np.asarray(alpha)>0
        h=np.array(chart["H"])
        new_h,diagnostic=fit(rgb,mask,h)
        planes=read(src/"sky"/name/"source-handoff/surfaces.json")["planes"]
        outline_h=outline_orientation(h,rgb,planes,chart["cells"])
        roots={"before":a/"assembly"}
        for method, matrix in (("line-fit",new_h),("outline-fit",outline_h)):
            out=RUN/panel/method
            if not np.allclose(matrix,h):
                replacement=extend(chart,matrix,mask,rgb)
                replacement["confidence"]=chart["confidence"]
            else:
                replacement=chart
            save(out/"charts"/rel,dict(raw,charts=[replacement if c["id"]==wall else c for c in raw["charts"]]))
            with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
                combined.run((panel,case,out/"charts",out/"assembly"))
            roots[method]=out/"assembly"
        scores=summarize([case],roots,str(LOCATION))["rows"][0]
        results[name]=dict(wall=wall,lines=diagnostic,scores=scores)
        print(name,scores["paired"]["outline-fit"],flush=True)
    save(OUT/"line-shear.json.gz",results)


if __name__=="__main__":
    run()
