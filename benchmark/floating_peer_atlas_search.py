"""Find peer-sized complete atlas windows before validating repeated source faces across the wall."""
from collections import Counter

import cv2
import numpy as np

from benchmark.floating_peer_ruler_refit import (RES, candidates, harmonic_margin, perspective, profiles,
    project, texture)
from benchmark.floating_verified_wall_expansion import full_face
from benchmark.full_block_lattice import atlas
from engine.atlas_count import _shortlist


def search(chart,ratios,mask,rgb):
    old=np.asarray(chart["H"])
    xy=np.column_stack(np.nonzero(mask)[::-1])+.5
    split=np.quantile(xy[:,0],[.4,.6])
    pivot=np.median(project(np.linalg.inv(old),xy),axis=0)
    h=perspective(old,pivot,1.)
    uv=project(np.linalg.inv(h),xy)
    lo=np.floor(uv.min(0))-1
    shape=np.ceil((uv.max(0)-lo+1)*RES).astype(int)
    yy,xx=np.indices(shape[::-1])
    points=project(h,np.stack((xx,yy),-1)/RES+lo)
    m=(points-.5).astype("f")
    crop=cv2.remap(rgb,m,None,cv2.INTER_LINEAR)
    support=cv2.remap(mask.astype("B"),m,None,cv2.INTER_NEAREST).astype(bool)&(points[...,0]<split[0])
    field=crop.mean(2).astype("f")
    integral=cv2.integral(support.astype("B"))
    library=atlas()
    votes=Counter()
    for ys in np.array_split(np.arange(len(crop)),3):
        for xs in np.array_split(np.arange(crop.shape[1]),3):
            region=np.ix_(ys,xs)
            kept=support[region]
            if kept.sum()>=25:
                shortlist,prior=_shortlist(crop[region],kept,library)
                for i,key in enumerate(shortlist[:4]):
                    votes[int(key)]+=1/(1+i)
    keys=[k for k,v in votes.most_common(6)]
    # The original wall provides foreshortening. The peer-height bracket supplies both
    # starting pitches; horizontal width may differ by a quarter from the height.
    heights=np.unique(np.rint(np.linspace(.75*min(ratios),1.25*max(ratios),9)*RES).astype(int))
    offers=[]
    for ht in heights:
        for width in np.unique(np.rint(ht*np.array([.75,1.,1.25])).astype(int)):
            if ht>field.shape[0] or width>field.shape[1]:
                continue
            valid=(integral[ht:,width:]-integral[:-ht,width:]-integral[ht:,:-width]+integral[:-ht,:-width])==ht*width
            if not valid.any():
                continue
            response=np.full(valid.shape,-2.,dtype='f')
            for key in keys:
                for orientation in range(8):
                    template=cv2.resize(library.patterns[key,orientation],(int(width),int(ht)),interpolation=cv2.INTER_LINEAR)
                    similarity=cv2.matchTemplate(field,template,cv2.TM_CCOEFF_NORMED)
                    np.maximum(response,similarity,out=response)
            response[~valid]=-2.
            for _ in range(2):
                _,corr,_,(x,y)=cv2.minMaxLoc(response)
                if corr<.35:
                    break
                # Pixel centers in the rectified view are at lo + i/resolution.
                start=lo+(np.array([x,y])-.5)/RES
                u,v=width/RES,ht/RES
                candidate=h@np.array([[u,0,start[0]],[0,v,start[1]],[0,0,1.]])
                proof=full_face(candidate,(0,0),mask,rgb)
                offers.append((proof.get("correlation",-1)+.3*proof.get("phase_advantage",-1),
                               float(corr),candidate,(u,v),start,proof))
                response[max(0,y-ht//2):y+ht//2+1,max(0,x-width//2):x+width//2+1]=-2
    if not offers:
        return None,dict(accepted=False,reason="no complete candidate window")
    score,match,winner,pitches,start,proof=max(offers,key=lambda s:(s[5]["ok"],s[0]))
    far=texture(winner,mask,rgb,split)[1]
    atlas_hits=[int(proof["ok"]),sum(s["ok"] for s in far)]
    prof=profiles(h,mask,rgb,split)
    tests=[dict(pitch=float(p),phase=float(phase),**harmonic_margin(profile,p,phase)) for p,phase,profile in zip(pitches,start,prof)]
    failures=[]
    if min(atlas_hits)==0:
        failures.append("whole atlas face in both partitions")
    if not all(a["validation"]>=.06 and a["margin"]>=.015 for a in tests):
        failures.append("distant two-axis and harmonic challenge")
    return winner,dict(accepted=not failures,failures=failures,atlas_hits=atlas_hits,axes=tests,
                       proposal_count=len(offers),near_verified=sum(s[5]["ok"] for s in offers),
                       highest_template_match=match,near=proof,far=far,perspective=1.,homography=winner.tolist())


if __name__=="__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from unittest.mock import patch
    from benchmark import floating_peer_ruler_refit as edge
    def main():
        p=argparse.ArgumentParser()
        p.add_argument("--report",action="store_true")
        args=p.parse_args()
        items=edge.work()
        if args.report:
            with patch.object(edge,"OUT",edge.OUT/"atlas-first"),patch.object(edge,"REVIEW",edge.REVIEW/"atlas-first"):
                edge.report(items)
        else:
            with ProcessPoolExecutor(5) as pool:
                list(pool.map(run_atlas,items))
    # Worker is imported by name under spawn.
    from benchmark.floating_peer_atlas_search import run_atlas
    main()


def run_atlas(job):
    from unittest.mock import patch
    from benchmark import floating_peer_ruler_refit as edge
    with patch.object(edge,"OUT",edge.OUT/"atlas-first"),patch.object(edge,"search",search):
        edge.run(job)
