"""Evaluate checking the current face before finding and substituting another complete face."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from benchmark.floating_one_face_control import WORKTREES,OUT,REVIEW,PANELS,inputs,base,combined,read,save,digest,summarize
from benchmark.floating_two_face_consistency import square_seed


def run(job):
    group,case=job
    key={"holdout3":"new6","holdout4":"new7"}.get(group,group)
    finder=WORKTREES/"floating-face-proposals/runs/face-proposals"/key
    verifier=WORKTREES/"floating-face-verifier/runs/face-verifier"/group
    _,original,whole,_=inputs(group)
    n=case["id"]
    rel=Path("sky")/n/"charts.json"
    packet=read(whole/rel)
    ordinary={c["id"]:c for c in read(original/rel)["charts"]}
    passes=read(verifier/"evidence"/(n+".json"))
    assert passes["expanded_sha256"]==digest(whole/rel)
    ext={r["id"]:r["extend"] for r in passes["charts"]}
    discovered=read(finder/"charts"/rel)
    assert discovered["source_packet_sha256"]==digest(whole/rel)
    offers={r["id"]:r["picked"] for r in read(finder/"offers"/(n+".json.gz"))["charts"]}
    fallback=read(verifier/"charts"/rel)
    results={arm:[] for arm in ("verify-then-find","repair-only")}
    for old,new,conservative in zip(packet["charts"],discovered["charts"],fallback["charts"],strict=True):
        cid=old["id"]
        assert cid==new["id"]==conservative["id"]
        replace=offers.get(cid) is not None and (not ext.get(cid,True) or not square_seed(ordinary[cid],new))
        results["verify-then-find"].append(new if replace else conservative)
        results["repair-only"].append(new if replace else old)
    for arm,charts in results.items():
        loc=OUT/group/arm
        save(loc/"charts"/rel,dict(packet,charts=charts,verifier_input=digest(verifier/"charts"/rel),
                                   proposal_input=digest(finder/"charts"/rel)))
        with patch.object(combined,"inputs",inputs):
            combined.run((group,case,loc/"charts",loc/"assembly"))


if __name__=="__main__":
    groups=(*PANELS,"holdout3","holdout4")
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(run,[(g,c) for g in groups for c in inputs(g)[3]]))
    results={}
    for g in groups:
        paths={"horizontal":base(g),**{a:OUT/g/a/"assembly" for a in ("verify-then-find","repair-only")}}
        results[g]=summarize(inputs(g)[3],paths,"connected-atlas/sky")
        print(g,[round(v["geometry_strict"]["f1"]*100,2) for v in results[g]["cohorts"]["all"].values()])
    save(REVIEW/"sequence-all.json.gz",results)
