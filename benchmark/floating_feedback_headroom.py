"""Exact pooled strict-F1 chooser across already decoded complete exports."""
from collections import Counter

from benchmark.floating_extra_feedback import WORKTREES, read
from benchmark.floating_reviewed_rulers import OUT
from benchmark.expanded_baseline import save


def objective(score):
    hits = score["geometry"]["strict"]
    return 2*hits["correct"], 2*hits["correct"]+hits["missed"]+hits["extras"]


def best(options):
    # Fractional programming with integral comparisons and an exact termination test.
    num, den = 0, 1
    while True:
        choices = []
        for arms in options:
            def merit(name):
                u,v = objective(arms[name])
                return u*den-v*num
            choices.append(max(arms,key=merit))
        uv = [objective(a[c]) for a,c in zip(options,choices)]
        u,v = map(sum,zip(*uv))
        if u*den==num*v:
            return choices
        num,den = u,v


def main():
    def rows(branch,folder,filename):
        return read(WORKTREES/branch/"benchmark/reviews"/folder/filename)
    audit = rows("floating-feedback-audit","floating-feedback-audit-01","decoded.json.gz")["rows"]
    scale = {(r["panel"],r["id"]):r for r in rows("floating-feedback-scale","floating-feedback-scale-01","conditions.json.gz")}
    support = {(r["group"],r["id"]):r for r in rows("floating-feedback-support","floating-feedback-support-01","paired.json.gz")}
    report={}
    for name,items in (("all92",audit),("sky",[r for r in audit if r["panel"]=="sky"]),
                       ("transfer",[r for r in audit if r["feedback_split"]=="other-archived-layout"])):
        choices = []
        for r in items:
            k=(r["panel"],r["id"])
            opts={a:r["scores"][a] for a in ("large-contraction","floating-default","baseline","source-one")}
            for row,arm in [(scale[k],"add-atlas"),(support[k],"sparse"),(support[k],"smooth")]:
                assert row["source_sha256"]==r["source_sha256"]
                assert objective(row["score"]["large-contraction"])==objective(opts["large-contraction"])
                opts[arm]=row["score"][arm]
            choices.append(opts)
        report[name]={}
        for group,opts in (("gate-only",[{k:a[k] for k in ("large-contraction",)} for a in choices]),
                           ("gate-or-sparse-default",[{k:a[k] for k in ("large-contraction","floating-default")} for a in choices]),
                           ("all-seven",choices)):
            picks=best(opts)
            totals={}
            for field in ("geometry","scores"):
                for metric in ("strict","facade"):
                    t=Counter()
                    for a,pick in zip(opts,picks):
                        t.update({k:a[pick][field][metric][k] for k in ("correct","missed","extras")})
                    t["f1"]=2*t["correct"]/(2*t["correct"]+t["missed"]+t["extras"])
                    totals[field+"_"+metric]=dict(t)
            report[name][group]=dict(totals=totals, picks=Counter(picks),
                                    selected=[(r["panel"],r["id"],p) for r,p in zip(items,picks)])
            print(name,group,totals["geometry_strict"],flush=True)
    save(OUT/"export-chooser.json.gz",report)


if __name__=="__main__":
    main()
