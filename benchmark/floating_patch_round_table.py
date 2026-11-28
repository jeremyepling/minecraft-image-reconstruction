"""Publish comparable end-to-end scores for every source-only patch variation."""
from collections import Counter
from benchmark.expanded_baseline import read, WORKTREES
from benchmark.floating_patch_report import COHORTS,REVIEW


if __name__=="__main__":
    groups=(*COHORTS,"holdout2")
    lines=["# Source-only comparisons", "",
           "Every row uses the floating assembler with horizontal faces enabled. Scores pool correct, missed and extra blocks across each subset before calculating F1. A single global translation is used for each export.", "",
           "| Panel | Cohort | Method | Placement | Facade placement | Placement + ID | Facade placement + ID |",
           "|---|---|---|---:|---:|---:|---:|"]
    for g in groups:
        name={"holdout":"new","holdout2":"new2"}.get(g,g)
        lattice=read(WORKTREES/"floating-patch-lattice/benchmark/reviews/floating-patch-lattice-01"/(name+".json.gz"))
        suffix="-holdout2" if g=="holdout2" else ""
        depth=read(REVIEW/("planes-horizontal"+suffix+".json.gz"))[g]
        opposed=read(REVIEW/("opposed-horizontal"+suffix+".json.gz"))[g]
        suffix="holdout2" if g=="holdout2" else "-".join(COHORTS)
        agreement=read(REVIEW/("consensus-"+suffix+".json.gz"))[g]
        scores={}
        def add(group,method,record):
            target=scores.setdefault((group,method),{})
            for field in ("geometry","scores"):
                for metric in ("strict","facade"):
                    stat=record[field][metric]
                    target.setdefault((field,metric),Counter()).update({k:stat[k] for k in ("correct","missed","extras")})
        indexes=[{r["id"]:r for r in src["rows"]} for src in (depth,opposed,agreement)]
        for a in lattice["frames"]:
            b,c,d=(idx[a["id"]] for idx in indexes)
            assert a["scores"]["horizontal"]["geometry"]["strict"]["correct"]==b["score"]["horizontal"]["geometry"]["strict"]["correct"]
            for cohort in ("all",a["dataset"]):
                for label,arm in (("Current floating candidate","horizontal"),("Local lattice veto","trim-horizontal"),("Local lattice refit","rechart-horizontal")):
                    add(cohort,label,a["scores"][arm])
                add(cohort,"Local plane veto",b["score"]["planes-horizontal"])
                add(cohort,"Both vetoes required",d["score"]["consensus"])
                add(cohort,"Contradiction required",c["score"]["opposed-horizontal"])
        for (cohort,label),metrics in scores.items():
            values=[f'{200*s["correct"]/(2*s["correct"]+s["missed"]+s["extras"]):.2f}' for s in metrics.values()]
            lines.append("| "+" | ".join([g,cohort,label,*values])+" |")
    (REVIEW/"source-results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
