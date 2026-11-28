"""Retain whole-wall cells unless both source-only patch tests independently remove them."""
import argparse
from concurrent.futures import ProcessPoolExecutor

from benchmark.floating_patch_combined import run
from benchmark.floating_patch_report import COHORTS, baseline, collect, write
from benchmark.floating_patch_controls import inputs, OUT, WORKTREES, read, save, digest


def build(job):
    group,case=job
    name=case["id"]
    key={"holdout":"new","holdout2":"new2"}.get(group,group)
    rel="sky/"+name+"/charts.json"
    source=inputs(group)[2]/rel
    packet=read(source)
    path1=WORKTREES/"floating-patch-lattice/runs/patch-lattice"/key/"trim/charts"/rel
    path2=WORKTREES/"floating-patch-planes/runs/patch-planes"/key/"planes/charts"/rel
    def chart_cells(path):
        data=read(path)
        for field in ("source_sha256","surfaces_sha256","counts_sha256"):
            assert data[field]==packet[field]
        return {c["id"]:c for c in data["charts"]}
    left,right=chart_cells(path1),chart_cells(path2)
    output=[]
    for c in packet["charts"]:
        keep=set()
        for variant in (left,right):
            v=variant[c["id"]]
            assert v["H"]==c["H"]
            keep.update((cell["u"],cell["v"]) for cell in v["cells"])
        cells=[cell for cell in c["cells"] if (cell["u"],cell["v"]) in keep]
        output.append(dict(c,cells=cells))
    root=OUT/group/"consensus-horizontal"
    save(root/"charts"/rel,dict(packet,charts=output,consensus_inputs=[digest(path) for path in (source,path1,path2)]))
    run((group,case,root/"charts",root/"assembly"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups",nargs="*",default=COHORTS)
    groups=parser.parse_args().groups
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(build,[(g,c) for g in groups for c in inputs(g)[3]]))
    write("consensus-"+"-".join(groups),{g:collect(g,{"horizontal":baseline(g,True),
          "consensus":OUT/g/"consensus-horizontal/assembly"}) for g in groups})
