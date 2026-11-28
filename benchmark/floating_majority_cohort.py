"""Apply the frozen upper-row treatments to every vertical chart on the 84 active conditions."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from benchmark.floating_majority_edge import (
    ISOLATION, LOCATION, REVIEW, RUN, WORKTREES, Image, Path, _native_alpha, combined,
    digest, np, patch, paths, read, save, source_rule, summarize,
)
from benchmark.floating_visible_row import extend


def run(job):
    row=job
    panel,name=row["panel"],row["id"]
    root=RUN/"cohort"/panel
    base=ISOLATION/panel/"large-contraction"
    rel=Path("sky")/name/"charts.json"
    packet=read(base/"charts"/rel)
    (source,_,_,cases),_,_=paths(panel)
    case=next(c for c in cases if c["id"]==name)
    path=Path(case["variants"]["sky"]["image"])
    assert digest(path)==row["source_sha256"]
    image=np.array(Image.open(path).convert("RGB"))
    planes={p["id"]:p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    arms={name:[] for name in ("prior","majority","seam")}
    for c in packet["charts"]:
        result=[c,c,c]
        if c["cells"] and c.get("normal_camera") and abs(c["normal_camera"][1])<.67:
            alpha,(x,y,r,b)=_native_alpha(planes[c["id"]]["support"],image.shape[1::-1])
            mask=np.zeros(image.shape[:2],bool)
            mask[y:b,x:r]=np.array(alpha)>0
            prior,proposals=extend(c,mask,image,-1.)
            majority,seam,_=source_rule(c,proposals,mask,image)
            result=[prior,majority,seam]
        for key,new in zip(arms,result):
            arms[key].append(new)
    roots={"previous":base/"assembly"}
    for key,charts in arms.items():
        out=root/key
        save(out/"charts"/rel,dict(packet,charts=charts))
        # Reuse only when the entire input packet is identical.
        roots[key]=next((old_root for old,old_root in roots.items() if
            (packet["charts"] if old=="previous" else arms[old])==charts),None)
        if roots[key] is None:
            with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
                combined.run((panel,case,out/"charts",out/"assembly"))
            roots[key]=out/"assembly"
    scores=summarize([case],roots,str(LOCATION))["rows"][0]
    result=dict(panel=panel,split=case["split"],source_sha256=row["source_sha256"],input_sha256=digest(base/"charts"/rel),
                scores=scores,counts={k:sum(len(c["cells"]) for c in a) for k,a in dict(previous=packet["charts"],**arms).items()})
    save(root/"cases"/(name+".json.gz"),result)
    return result


def main():
    index=WORKTREES/"floating-textured-cohort/benchmark/reviews/floating-textured-cohort-01/scores.json.gz"
    rows=[r for r in read(index)["rows"] if r["active"]]
    with ProcessPoolExecutor(5) as pool:
        results=list(pool.map(run,rows))
    stats={}
    for row in results:
        for arm,score in row["scores"]["score"].items():
            for group in ("all",row["split"]):
                groupstats=stats.setdefault(group,{}).setdefault(arm,{})
                for field in ("geometry","scores"):
                    for kind in ("strict","facade"):
                        met=groupstats.setdefault(field+"_"+kind,Counter())
                        met.update({k:score[field][kind][k] for k in ("correct","missed","extras")})
    for group in stats.values():
        for arm in group.values():
            for v in arm.values():
                v["f1"]=2*v["correct"]/(2*v["correct"]+v["missed"]+v["extras"])
    save(REVIEW/"cohort.json.gz",dict(active_manifest_sha256=digest(index),rows=results,stats=stats))


if __name__=="__main__":
    main()
