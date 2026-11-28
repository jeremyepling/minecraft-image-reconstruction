"""Audit V1 decoded whole packs, reference IDs, source provenance and same-image transfer."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

from benchmark import floating_v1_run as v1
from benchmark.expanded_baseline import digest,read,save
from benchmark.floating_distant_seams import data as historic_data
from benchmark.mask_blocks import decoded_pack

REVIEW=v1.ROOT/"benchmark/reviews/floating-visible-builds-v1-01"
COHORTS=("all","fixed-intersection","new-to-84","seen-in-84","train","validation","test","regression")
METRICS=("geometry","scores")
ARMS=("sparse","split-border","current")


def aggregate(rows,strict_orientation=False):
    result={}
    for mode in ARMS:
        totals={}
        for metric in METRICS:
            for kind in ("strict","facade"):
                counts=Counter()
                for r in rows:
                    arm=r["arms"][mode]
                    item=arm["score"][metric][kind]
                    if strict_orientation and arm["nonempty_unresolved"]:
                        counts.update(dict(correct=0,missed=item["correct"]+item["missed"],extras=arm["exported"]))
                    else:
                        counts.update({k:item[k] for k in ("correct","missed","extras")})
                tp,fn,fp=[counts[k] for k in ("correct","missed","extras")]
                totals[f"{metric}_{kind}"]=dict(counts,f1=2*tp/(2*tp+fn+fp) if rows else None)
        total=lambda k:sum(r["arms"][mode][k] for r in rows)
        result[mode]=dict(totals,empty=total("empty"),nonempty_unresolved=total("nonempty_unresolved"),count=len(rows))
    return result


def analyze(case,row,old):
    n=case["id"]
    roots={"sparse":v1.OUT/"sparse/assembly","split-border":Path(row["roots"]["adopted"]),
           "current":Path(row["roots"]["nonshrinking"])}
    truth={tuple(b["xyz_world"]):b["block_id"] for b in read(Path(case["frame"])/"labels.json")["blocks"]}
    arms={}
    aligned={}
    overlap={}
    for mode,root in roots.items():
        folder=root/v1.PREFIX/n
        rec=read(folder/"score.json")
        pack=folder/"reconstruction.zip"
        assert rec["source_sha256"]==case["rgb_sha256"] and rec["reference_sha256"]==case["reference_sha256"]
        assert digest(pack)==rec["pack_sha256"]
        coords=read(folder/"placements.json")
        local=decoded_pack(pack)
        assert local=={tuple(p):m for p,m in coords["local"]}
        aligned[mode]={tuple(p):m for p,m in coords["aligned"]}
        score=rec["score"]["fixed"] or rec["score"]["relaxed_yaw"]
        arms[mode]=dict(score=score,exported=len(local),empty=not local,nonempty_unresolved=bool(local) and rec["score"]["fixed"] is None,
                        pack_sha256=rec["pack_sha256"],chart_sha256=rec["chart_sha256"],
                        selected=rec["selected_origins"],pack=str(pack))
        if old:
            src=historic_data(old["panel"])[0]
            other=src/"assembly" if mode=="sparse" else Path(old["roots"]["adopted" if mode=="split-border" else "nonshrinking"])
            other=other/v1.PREFIX/n
            assert (other/"score.json").is_file(),other
            receipt=read(other/"score.json")
            assert receipt["source_sha256"]==rec["source_sha256"] and receipt["reference_sha256"]==rec["reference_sha256"]
            overlap[mode]=dict(local_equal=local==decoded_pack(other/"reconstruction.zip"),
                               registered_equal=coords==read(other/"placements.json"),old_pack=receipt["pack_sha256"])
    def deltas(before,after):
        a,b=aligned[before],aligned[after]
        correct=set(truth)
        e,f=[{p for p,m in v.items() if truth.get(p)==m} for v in (a,b)]
        return dict(gain=len((b.keys()-a.keys())&correct),loss=len((a.keys()-b.keys())&correct),
                    add=len((b.keys()-a.keys())-correct),remove=len((a.keys()-b.keys())-correct),
                    exact_gain=len(f-e),exact_loss=len(e-f))
    return dict(id=n,dataset=case["dataset"],split=case["split"],layout=case["layout"],
                fixed_id=case["fixed_104_id"],seen_panel=old and old["panel"],same_image=overlap,
                source_sha256=case["rgb_sha256"],reference_sha256=case["reference_sha256"],
                fields=row["source_fields"],arms=arms,changed=roots["current"]!=roots["split-border"],
                package=dict(edge=row["edge"],corner=row["pair"],contraction=row["contracted"],pink=row["split"]),
                choices=row["source_choices"],paired=deltas("split-border","current"),sparse_delta=deltas("sparse","current"))


def report(partial=False):
    historic=read(v1.ROOT/"benchmark/reviews/floating-pregrid-depth-refit-01/conditions.json.gz")
    seen={r["source_sha256"]:r for r in historic}
    rows=[]
    for c in v1.tasks():
        path=v1.OUT/"rows"/(c["id"]+".json.gz")
        if not path.exists() and partial:
            continue
        rows.append(analyze(c,read(path),seen.get(c["rgb_sha256"])))
    groups={"all":rows,"fixed-intersection":[r for r in rows if r["fixed_id"]],
            "new-to-84":[r for r in rows if not r["seen_panel"]],"seen-in-84":[r for r in rows if r["seen_panel"]]}
    groups.update({k:[r for r in rows if r["split"]==k] for k in ("train","test","validation","regression")})
    totals={k:aggregate(v) for k,v in groups.items()}
    print(len(rows),{k:v["geometry_strict"] for k,v in totals["all"].items()},flush=True)
    if partial:
        return
    save(REVIEW/"conditions.json.gz",rows)
    save(REVIEW/"summary.json",totals)
    save(REVIEW/"strict-orientation.json",{k:aggregate(v,True) for k,v in groups.items()})
    import gzip
    if v1.FIXTURES.exists():
        (REVIEW/"fixtures.json.gz").write_bytes(gzip.compress(v1.FIXTURES.read_bytes(),mtime=0))
    evidence=[read(v1.OUT/"no-op/rows"/(r["id"]+".json")) for r in rows]
    assert all(v["decoded_equal"] and v["registered_equal"] for v in evidence)
    save(REVIEW/"no-op.json.gz",evidence)
    with ZipFile(REVIEW/"exports.zip","w",compression=ZIP_DEFLATED) as z:
        for r in rows:
            for arm,v in r["arms"].items():
                entry=ZipInfo(r["id"]+"/"+arm+".zip")
                entry.compress_type=ZIP_DEFLATED
                z.writestr(entry,Path(v["pack"]).read_bytes())
    with ZipFile(REVIEW/"exports.zip") as z:
        import hashlib
        assert all(hashlib.sha256(z.read(r["id"]+"/"+arm+".zip")).hexdigest()==v["pack_sha256"]
                   for r in rows for arm,v in r["arms"].items())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--partial",action="store_true")
    report(p.parse_args().partial)
