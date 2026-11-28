"""Independently decode each placement export and pool the paired visible-only metrics."""
from benchmark.floating_bridges_report import summarize
from benchmark.floating_patch_controls import inputs, OUT,SKY,ROOT,ARMS,WORKTREES,save

REVIEW=ROOT/"benchmark/reviews/floating-patch-controls-01"
COHORTS=("original","fresh","supplemental","sky","holdout")


def baseline(cohort,horizontal=False):
    if horizontal:
        if cohort=="sky":
            return SKY/"horizontal-cells"
        if cohort=="holdout":
            return inputs(cohort)[2].parents[1]/"horizontal-cells/assembly"
        return WORKTREES/"floating-horizontal-bridges/runs/horizontal-cells/assembly"
    if cohort=="sky":
        return SKY/"whole-wall"
    return inputs(cohort)[2].parent/"assembly"


def collect(cohort,roots):
    return summarize(inputs(cohort)[3],roots,"connected-atlas/sky")


def write(name, results):
    lines=["# "+name.replace("-"," ").title(),"",
           "F1 percentages are pooled across blocks. Gained and lost positions are compared after independently registering each whole build. The four score columns give strict position, facade position, strict position and material, and facade position and material.","",
           "| Panel | Cohort | Method | Position | Facade | Position + ID | Facade + ID | Correct / missed / extra | Gained / lost / extras added / extras removed |",
           "|---|---|---|---:|---:|---:|---:|---|---|"]
    for panel,data in results.items():
        for group,arms in data["cohorts"].items():
            for arm,s in arms.items():
                vals=[f'{100*s[k]["f1"]:.2f}' for k in ("geometry_strict","geometry_facade","scores_strict","scores_facade")]
                counts=" / ".join(str(s["geometry_strict"][k]) for k in ("correct","missed","extras"))
                diff=" / ".join(str(s["paired"][k]) for k in ("gained","lost","added","removed"))
                lines.append("| "+" | ".join((panel,group,arm,*vals,counts,diff))+" |")
    save(REVIEW/(name+".json.gz"),results)
    (REVIEW/(name+".md")).write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    results={}
    for cohort in COHORTS:
        roots={"whole-wall":baseline(cohort)}
        roots.update({a:OUT/cohort/a/"assembly" for a in ARMS})
        results[cohort]=collect(cohort,roots)
    write("reference-controls",results)
