"""Collate decoded placement evidence and choose among measured exports for a diagnostic ceiling."""
from collections import Counter
from benchmark.floating_one_face_control import PANELS,REVIEW,WORKTREES,read,save

FIELDS=[("geometry","strict"),("geometry","facade"),("scores","strict"),("scores","facade")]
NAMES={"horizontal":"Current floating candidate","source-one":"Source-discovered face","pair-required":"Require two discovered faces",
       "pair-refined":"Refine with a second face if available","square-challenge":"Replace a non-square seed",
       "one-view":"One exact face in one chart per view","one-chart":"One exact face per eligible chart",
       "one-chart-own-plane":"Exact face and visible surface","pixel-center":"One face with rounded corners",
       "two-pixel-center":"Two rounded faces with known offset","prior-perspective":"Rounded face with source-estimated perspective",
       "geometry-chooser":"Perfect chooser among source exports","verified":"Verify existing seeds","logistic":"Learned seed verifier",
       "repair-only":"Verify before substituting","verify-then-find":"Verify or find, fall back to ordinary seed extent",
       "native-repair":"Correct only native-unqualified charts using proposed faces"}


def best(rows,arms):
    # Maximize micro F1 across the panel, keeping geometry-selected registrations and IDs.
    rate=0.
    for _ in range(24):
        hits=den=0
        choices=[]
        for row in rows:
            values=[]
            for arm in arms:
                s=row[arm]["geometry"]["strict"]
                d=2*s["correct"]+s["missed"]+s["extras"]
                values.append((2*s["correct"]-rate*d,arm,s["correct"],d))
            _,arm,h,d=max(values,key=lambda v:v[0])
            choices.append(arm)
            hits+=h
            den+=d
        new=2*hits/den
        if abs(rate-new)<1e-12:
            return choices
        rate=new
    raise RuntimeError("Pooled chooser did not converge")


def main():
    cases={}
    panels=(*PANELS,"holdout3","holdout4")
    for pattern in ("oracle-*","precision-*","stabilize-*","pairs-*","shape-*","sequence-*"):
        for path in REVIEW.glob(pattern+".json.gz"):
            for group,record in read(path).items():
                panel=cases.setdefault(group,{})
                for row in record["rows"]:
                    ds,arms=panel.setdefault(row["id"],(row["dataset"],{}))
                    assert ds==row["dataset"]
                    for arm,score in row["score"].items():
                        if arm in arms:
                            assert arms[arm]==score
                        arms[arm]=score
    repair=WORKTREES/"floating-face-proposals/benchmark/reviews/floating-face-proposals-01/reference-repair.json.gz"
    for key,record in read(repair).items():
        group={"new6":"holdout3","new7":"holdout4"}.get(key,key)
        for row in record["rows"]:
            choices=[s for a,s in row["score"].items() if a!="horizontal"]
            cases[group][row["id"]][1]["native-repair"]=choices[-1]
    verf=WORKTREES/"floating-face-verifier/benchmark/reviews/floating-face-verifier-01/placements.json.gz"
    for group,record in read(verf).items():
        for row in record["rows"]:
            arms=cases[group][row["id"]][1]
            for arm,score in row["score"].items():
                if arm=="horizontal":
                    assert arms[arm]["geometry"]==score["geometry"]
                elif arm in ("verified","logistic"):
                    arms[arm]=score
    summary={}
    for oracle in (False,True):
        filename="reference-scores" if oracle else "source-scores"
        names=["horizontal",*(["one-view","one-chart","pixel-center","prior-perspective","two-pixel-center","one-chart-own-plane","native-repair"]
            if oracle else ["source-one","square-challenge","pair-required","pair-refined","verified","logistic","repair-only","verify-then-find","geometry-chooser"])]
        lines=["# "+filename.replace("-"," ").capitalize(),"","The four F1 columns are percentages. Each whole structure is registered once before scoring.","",
               "| Panel | Dataset | Method | Strict placement | Facade placement | Strict position + ID | Facade position + ID | Correct / missed / extra |",
               "|---|---|---|---:|---:|---:|---:|---|"]
        for g in panels:
            if g not in cases:
                continue
            panel=cases[g]
            rows=[r for _,r in panel.values()]
            if not oracle:
                options=[a for a in names if a!="geometry-chooser" and all(a in r for r in rows)]
                for row,arm in zip(rows,best(rows,options)):
                    row["geometry-chooser"]=row[arm]
            for ds in ("all",*sorted(set(k for k,_ in panel.values()))):
                entries=[r for k,r in panel.values() if ds=="all" or ds==k]
                for arm in names:
                    if not all(arm in r for r in entries):
                        continue
                    values=[]
                    for field,metric in FIELDS:
                        s=Counter()
                        for r in entries:
                            q=r[arm][field][metric]
                            s.update({k:q[k] for k in ("correct","missed","extras")})
                        f1=200*s["correct"]/(2*s["correct"]+s["missed"]+s["extras"])
                        values.append(f"{f1:.2f}")
                        if field=="geometry" and metric=="strict":
                            counts=" / ".join(str(s[k]) for k in ("correct","missed","extras"))
                    lines.append("| "+" | ".join([g,ds,NAMES.get(arm,arm),*values,counts])+" |")
                    if ds=="all":
                        summary.setdefault(g,{})[arm]=values
        (REVIEW/(filename+".md")).write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(REVIEW/"summary.json",summary)


if __name__=="__main__":
    main()
