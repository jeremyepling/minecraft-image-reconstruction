"""Compose two frozen source-only decisions on expanded charts and rerun ordinary assembly."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zipfile import ZipFile,ZIP_DEFLATED,ZipInfo
import hashlib
import json

from benchmark.expanded_baseline import ROOT,WORKTREES,digest,read,save
from benchmark.floating_verified_wall_expansion import BASE,PREFIX,jobs
from benchmark.floating_distant_seams import data,combined
from benchmark.floating_bridges_report import read_arm
from benchmark.mask_blocks import decoded_pack

OUT=ROOT/"runs/selected-package"
REVIEW=ROOT/"benchmark/reviews/floating-selected-package-01"
EDGE=WORKTREES/"floating-edge-aligned-ruler"
PAIR=WORKTREES/"floating-corner-pair-repair"
EDGE_MANIFEST=EDGE/"benchmark/reviews/floating-edge-aligned-ruler-01/conditions.json.gz"
PAIR_MANIFEST=PAIR/"benchmark/reviews/floating-corner-pair-01/rows.json.gz"
HASHES={"edge":"1b825705a210cac9c9dec284d673c42b3780badb0765377a80ada144a53822f0",
        "pair":"26b63fb03d3fd2a43562abc886f98a1d3738592676d8754a80670fed274d0953"}
REVISIONS={"edge":"a0e67a7","pair":"27e86b1","large_contraction":"e97dd44"}


def selected_inputs():
    assert digest(EDGE_MANIFEST)==HASHES["edge"] and digest(PAIR_MANIFEST)==HASHES["pair"]
    # Only the prior source decisions and source hashes reach the composing step.
    edges={}
    for r in read(EDGE_MANIFEST):
        accepted=[cid for cid,v in r["charts"].items() if
                  v["refined"]["local_repeat_check"]["accepted"] and all(a["accepted"] for a in v["lines"])]
        edges[r["panel"],r["id"]]=(accepted,r["source"],r["mask"])
    pairs={}
    for r in read(PAIR_MANIFEST):
        arm=r["choice"]
        ids=next((v["pair"] for v in r["options"] if v["arm"]==arm),[])
        assert (arm=="keep")== (not ids)
        pairs[r["panel"],r["id"]]=(arm,ids,r["source_sha256"],r["hashes"]["keep"])
    work=jobs()
    keys={(g,c["id"]) for g,c in work}
    assert keys==set(edges)==set(pairs) and len(keys)==len(work)==84
    return [(g,c,edges[g,c["id"]],pairs[g,c["id"]]) for g,c in work]


def materialize(job):
    g,c,edge,pair=job
    n=c["id"]
    rel=Path("sky")/n/"charts.json"
    original=BASE/g/"large-contraction/charts"/rel
    before=read(original)
    selected_edges,edge_source,mask_hash=edge
    pair_arm,selected_pair,pair_source,base_hash=pair
    assert digest(original)==base_hash
    assert digest(Path(c["variants"]["sky"]["image"]))==before["source_sha256"]==edge_source==pair_source
    assert digest(data(g)[0]/"sky"/n/"source-handoff/surfaces.json")==mask_hash
    # A future overlapping case must recheck pair validity against the changed
    # geometry. Both frozen selectors proved disjoint at condition level on this set.
    if selected_edges and selected_pair:
        raise ValueError(f"Overlapping selectors require a new pair validation on {g}/{n}")
    folder=BASE/g/"large-contraction"
    rule="large-contraction"
    ids=[]
    if selected_edges:
        folder=EDGE/"runs/edge-aligned-ruler"/g/"original-fallback"
        rule="edge-original-fallback"
        ids=selected_edges
    elif selected_pair:
        folder=PAIR/"runs/pair-repair"/g/pair_arm
        rule="corner-pair"
        ids=selected_pair
    chosen=folder/"charts"/rel
    packet=read(chosen)
    assert packet["source_sha256"]==edge_source
    old={v["id"]:v for v in before["charts"]}
    new={v["id"]:v for v in packet["charts"]}
    assert list(new)==list(old)
    assert {k for k in old if old[k]!=new[k]}==set(ids)
    dest=OUT/g/"charts"/rel
    dest.parent.mkdir(exist_ok=True,parents=True)
    copyfile(chosen,dest)
    chart_hash=digest(dest)
    assert chart_hash==digest(chosen)
    save(OUT/g/"handoffs"/(n+".json"),dict(panel=g,id=n,source_sha256=edge_source,surface_sha256=mask_hash,
         baseline_chart_sha256=base_hash,chart_sha256=chart_hash,rule=rule,changed_charts=ids,
         source_packet=str(chosen),expected_assembly=str(folder/"assembly"),source_revisions=REVISIONS))


def evaluate(job):
    g,c,_,_=job
    n=c["id"]
    receipt=read(OUT/g/"handoffs"/(n+".json"))
    with patch.object(combined,"inputs",data):
        combined.run((g,c,OUT/g/"charts",OUT/g/"assembly"))
    pack=OUT/g/"assembly"/PREFIX/n
    original=BASE/g/"large-contraction/assembly"/PREFIX/n
    expected=Path(receipt["expected_assembly"])/PREFIX/n
    after,after_world=read_arm(pack)
    before,before_world=read_arm(original)
    score=read(pack/"score.json")
    expected_score=read(expected/"score.json")
    baseline_score=read(original/"score.json")
    assert score["source_sha256"]==receipt["source_sha256"]
    assert score["chart_sha256"]==receipt["chart_sha256"]
    assert score["reference_sha256"]==expected_score["reference_sha256"]==baseline_score["reference_sha256"]
    assert decoded_pack(pack/"reconstruction.zip")==decoded_pack(expected/"reconstruction.zip")
    assert read(pack/"placements.json")==read(expected/"placements.json")
    assert score["score"]==expected_score["score"]
    truth={tuple(b["xyz_world"]):b["block_id"] for b in read(Path(c["frame"])/"labels.json")["blocks"]}
    visible={tuple(p["xyz_world"]) for p in read(Path(c["frame"])/"labels.json")["faces"]}
    def delta(old,new):
        return dict(gained=len((new.keys()-old.keys()) & visible),
                    lost=len((old.keys()-new.keys()) & visible),
                    added=len((new.keys()-old.keys()) - visible),
                    removed=len((old.keys()-new.keys()) - visible))
    def exact(blocks):
        return {p for p,b in blocks.items() if p in visible and truth[p]==b}
    was,now=exact(before_world),exact(after_world)
    assert len(was)==before["scores"]["strict"]["correct"] and len(now)==after["scores"]["strict"]["correct"]
    receipt.update(dataset=c["dataset"],split=c["split"],reference_sha256=score["reference_sha256"],
                   pack_sha256=digest(pack/"reconstruction.zip"),original_pack_sha256=digest(original/"reconstruction.zip"),
                   expected_pack_sha256=digest(expected/"reconstruction.zip"),baseline=before,package=after,
                   paired=delta(before_world,after_world),exact_gained=len(now-was),exact_lost=len(was-now),
                   selected_origins=score["selected_origins"],joins=score["joins"])
    save(OUT/g/"rows"/(n+".json.gz"),receipt)
    if receipt["changed_charts"]:
        print(g,n,receipt["rule"],receipt["paired"],receipt["exact_gained"],receipt["exact_lost"],flush=True)


def report(items):
    rows=[read(OUT/g/"rows"/(c["id"]+".json.gz")) for g,c,_,_ in items]
    save(REVIEW/"conditions.json.gz",rows)
    # Portable charts and actual Java datapacks, including all unchanged controls.
    # Keep published member names independent of local worktree paths.
    bundle=REVIEW/"selected-charts-and-packs.zip"
    with ZipFile(bundle,"w") as archive:
        for row in rows:
            n,g=row["id"],row["panel"]
            for folder,name,source,check in (
                    ("charts","charts.json",OUT/g/"charts/sky"/n/"charts.json",row["chart_sha256"]),
                    ("packs","reconstruction.zip",OUT/g/"assembly"/PREFIX/n/"reconstruction.zip",row["pack_sha256"])):
                content=source.read_bytes()
                assert hashlib.sha256(content).hexdigest()==check
                entry=ZipInfo("/".join((folder,g,n,name)),(1980,1,1,0,0,0))
                archive.writestr(entry,content,compress_type=ZIP_DEFLATED)
        manifest=[{k:r[k] for k in ("id","panel","dataset","split","source_sha256","surface_sha256",
                   "reference_sha256","baseline_chart_sha256","chart_sha256","pack_sha256","rule","changed_charts")}
                  for r in rows]
        archive.writestr(ZipInfo("manifest.json",(1980,1,1,0,0,0)),json.dumps(manifest,separators=(",",":")),
                         compress_type=ZIP_DEFLATED)
    with ZipFile(bundle) as archive:
        assert len(archive.infolist())==169 and archive.testzip() is None
    summary={}
    for key in ("all","sky",*sorted({r["dataset"] for r in rows})):
        subset=[r for r in rows if key in ("all",r["panel"],r["dataset"])]
        results={}
        for arm in ("baseline","package"):
            stats={}
            for f in ("geometry","scores"):
                for metric in ("strict","facade"):
                    count=Counter()
                    for row in subset:
                        s=row[arm][f][metric]
                        count.update({k:s[k] for k in ("correct","missed","extras")})
                    count["f1"]=2*count["correct"]/(2*count["correct"]+count["missed"]+count["extras"])
                    stats[f+"_"+metric]=dict(count)
            stats["assignment"]={k:sum(r[arm]["scores"][k] for r in subset)
                                 for k in ("assigned","fallback","assigned_correct","assigned_positions")}
            results[arm]=stats
        results.update(conditions=len(subset),rules=dict(Counter(r["rule"] for r in subset)),
            paired={k:sum(r["paired"][k] for r in subset) for k in ("gained","lost","added","removed")},
            exact_gained=sum(r["exact_gained"] for r in subset),exact_lost=sum(r["exact_lost"] for r in subset))
        summary[key]=results
    save(REVIEW/"summary.json",dict(revisions=REVISIONS,manifests=HASHES,bundle_sha256=digest(bundle),cohorts=summary))
    print(summary["all"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("charts","assembly","report","all"))
    p.add_argument("--workers",type=int,default=5)
    args=p.parse_args()
    inputs=selected_inputs()
    for stage,task in (("charts",materialize),("assembly",evaluate)):
        if args.stage in (stage,"all"):
            with ProcessPoolExecutor(args.workers) as pool:
                list(pool.map(task,inputs))
    if args.stage in ("report","all"):
        report(inputs)
