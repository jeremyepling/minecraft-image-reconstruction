"""Learn a full-face decision on three training layouts; score a separate native-label gate."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
import argparse

import cv2
import numpy as np

from benchmark.floating_face_verifier import GROUPS, OUT, REVIEW, WORKTREES, baseline, digest, floating_patch_combined, panel, read, save, summarize

FEATURES = ("coverage","log_contrast","heldout","boundary_margin","neighbor","train_score","tested")
TRAIN = {"watchpost2","observatory-02","riverside-mill-2"}
THRESHOLD = .70
AUDIT = WORKTREES/"floating-face-audit/benchmark/reviews/floating-face-audit-01/seed-audit.json.gz"


def features(s):
    return [s["coverage"],np.log1p(s.get("contrast",0)),s.get("heldout",0),
            s.get("boundary_margin",0),s.get("neighbor",-1),s.get("train_score",0),int("heldout" in s)]


def label(seed):
    return seed["classification"]=="aligned"


def sigmoid(logits):
    return 1/(1+np.exp(-np.clip(logits,-40,40)))


def fit_model(samples):
    x=np.asarray([s[0] for s in samples])
    y=np.asarray([s[1] for s in samples],float)
    layouts=np.array([s[2] for s in samples])
    if not (y.any() and (1-y).any()):
        raise ValueError("Training split contains only one class")
    # Give each training layout the same total weight.
    weight=np.array([len(y)/(len(set(layouts))*sum(layouts==g)) for g in layouts])
    mean=np.average(x,axis=0,weights=weight)
    scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weight)),.05)
    design=np.column_stack([np.ones(len(y)),(x-mean)/scale])
    reg=np.diag([0]+[4.]*(design.shape[1]-1))
    coef=np.zeros(design.shape[1])
    for _ in range(40):
        prob=sigmoid(design@coef)
        grad=design.T@(weight*(prob-y))+reg@coef
        hess=design.T@((weight*prob*(1-prob))[:,None]*design)+reg
        change=np.linalg.solve(hess,grad)
        coef-=change
        if np.linalg.norm(change)<1e-8:
            break
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),counts=[int(y.sum()),len(y)])


def prob(seed,model):
    x=(np.array(features(seed))-model["mean"])/model["scale"]
    return float(sigmoid(np.r_[1.,x]@model["coef"]))


def truth_index(case):
    return {(c["id"],s["u"],s["v"]):label(s) for c in case["charts"] for s in c["seeds"]}


def replay(job):
    group,case,truth,model=job
    cv2.setNumThreads(1)
    name=case["id"]
    rel=Path("sky")/name/"charts.json"
    _,ordinary,whole,_=panel(group)
    packet=read(whole/rel)
    seeds={c["id"]:c for c in read(ordinary/rel)["charts"]}
    evidence=read(OUT/group/"evidence"/(name+".json"))
    assert evidence["expanded_sha256"]==digest(whole/rel) and evidence["ordinary_sha256"]==digest(ordinary/rel)
    use={}
    rows=[]
    for c in evidence["charts"]:
        decisions=[]
        for s in c["seeds"]:
            aligned=truth[c["id"],s["u"],s["v"]]
            predicted=prob(s,model)
            decisions.append(dict(u=s["u"],v=s["v"],aligned=aligned,heuristic=s["accept"],
                                  logistic=predicted>=THRESHOLD,probability=predicted))
        use[c["id"]]={a:any(s[a] for s in decisions) for a in ("logistic","aligned")}
        rows.append(dict(id=c["id"],seeds=decisions))
    # Oracle runs only after source-only decisions are fixed; each arm uses the same assembler and chart options.
    for key,arm in (("logistic","logistic"),("aligned","oracle")):
        dest=OUT/group/arm
        charts=[seeds[c["id"]] if c["id"] in use and not use[c["id"]][key] else c for c in packet["charts"]]
        save(dest/"charts"/rel,dict(packet,charts=charts))
        with patch.object(floating_patch_combined,"inputs",panel):
            floating_patch_combined.run((group,case,dest/"charts",dest/"assembly"))
    return dict(group=group,id=name,layout=case["layout"],split=case["split"],charts=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups",nargs="*",default=[*GROUPS,"holdout3"])
    args=p.parse_args()
    audit=read(AUDIT)
    audits={(r["group"],r["id"]):truth_index(r) for r in audit["cases"]}
    training=[]
    for case in panel("supplemental")[3]:
        assert case["split"]=="train" and case["layout"] in TRAIN
        evidence=read(OUT/"supplemental/evidence"/(case["id"]+".json"))
        truth=audits["supplemental",case["id"]]
        training.extend((features(s),truth[c["id"],s["u"],s["v"]],case["layout"]) for c in evidence["charts"] for s in c["seeds"])
    models={name:fit_model([r for r in training if r[2]!=name]) for name in (*sorted(TRAIN),"all")}
    jobs=[(g,c,audits[g,c["id"]],models.get(c["layout"],models["all"])) for g in args.groups for c in panel(g)[3]]
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(replay,jobs))
    save(REVIEW/"classification.json.gz",dict(audit_sha256=digest(AUDIT),training=sorted(TRAIN),threshold=THRESHOLD,
                                              features=FEATURES,models=models,cases=rows))
    results={}
    for g in args.groups:
        prior=panel(g)[0].parent/"horizontal-cells/assembly" if g=="holdout3" else baseline(g,True)
        arms={"horizontal":prior,"verified":OUT/g/"assembly"}
        arms.update({a:OUT/g/a/"assembly" for a in ("logistic","oracle")})
        results[g]=summarize(panel(g)[3],arms,"connected-atlas/sky")
    save(REVIEW/"placements.json.gz",results)


if __name__=="__main__":
    main()
