"""Write pooled seed and decoded placement tables and retain scorer/atlas fingerprints."""
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from benchmark.relative_oracle_charts import UNIT
from engine.geometry import project
from benchmark.floating_face_verifier import OUT, REVIEW, atlas, baseline, digest, panel, read, save


def stats(cases, mode):
    truth=pred=tp=charts=accepted=valid=both=0
    for case in cases:
        for chart in case["charts"]:
            rows=chart["seeds"]
            correct=[r["aligned"] for r in rows]
            decisions=[r[mode] if mode else True for r in rows]
            truth+=sum(correct)
            pred+=sum(decisions)
            tp+=sum(t and p for t,p in zip(correct,decisions))
            charts+=1
            accepted+=any(decisions)
            valid+=any(correct)
            both+=any(correct) and any(decisions)
    return f"{tp}/{pred} ({100*tp/max(1,pred):.1f}%) | {tp}/{truth} ({100*tp/max(1,truth):.1f}%) | {both}/{accepted} | {both}/{valid}"


def overlay():
    width,height=440,260
    sheet=Image.new("RGB",(width*2,330),"#17232c")
    pen=ImageDraw.Draw(sheet)
    font=ImageFont.load_default(size=18)
    for i,(g,query) in enumerate((("sky","chapel"),("holdout3","stable-yard-1-warm-high"))):
        case=next(c for c in panel(g)[3] if query in c["id"])
        img=Image.open(case["variants"]["sky"]["image"]).convert("RGB")
        chart_data=read(panel(g)[1]/"sky"/case["id"]/"charts.json")
        seeds={c["id"]:c for c in chart_data["charts"]}
        evidence=read(OUT/g/"evidence"/(case["id"]+".json"))
        chart=next(c for c in evidence["charts"] if (i==0 or not c["extend"]))
        h=np.asarray(seeds[chart["id"]]["H"])
        polygons=[project(h,np.array([s["u"],s["v"]])+UNIT) for s in chart["seeds"]]
        xy=np.concatenate(polygons)
        center=(xy.min(0)+xy.max(0))/2
        unit=max(2.*np.ptp(xy,axis=0)[0]/width,2.*np.ptp(xy,axis=0)[1]/height,.20)
        origin=np.floor(center-np.array([width,height])*unit/2)
        end=origin+np.array([width,height])*unit
        crop=img.crop(tuple(np.r_[origin,end].astype(int))).resize((width,height),Image.Resampling.NEAREST)
        draw=ImageDraw.Draw(crop)
        for polygon,seed in zip(polygons,chart["seeds"]):
            points=[tuple((p-origin)/unit) for p in polygon]
            draw.line(points+points[:1],fill="#25efbc" if seed["accept"] else "#ff785a",width=2)
        col=i*width
        sheet.paste(crop,(col,32))
        pen.text((col+8,6),("Chapel: propagation retained","Stable Yard: ordinary extent retained")[i],font=font,fill="white")
    pen.text((8,300),"Green outlines passed the atlas rule. Orange outlines failed.",font=font,fill="white")
    sheet.save(REVIEW/"seed-decisions.jpg",quality=92)


def main():
    scores=read(REVIEW/"placements.json.gz")
    classification=read(REVIEW/"classification.json.gz")
    lines=["# Full-face verification before wall expansion","",
           "Keep this gate experimental. It measures whether a source chart contains at least one plausible full cube face. The ordinary extent is preserved when every seed fails. Qualifying a seed leaves the complete expansion and the horizontal assembler unchanged.","",
           "The rule was frozen at 0adb7d2. Fit an atlas texture and orientation on one native RGB pixel fold. On disjoint native pixels, verify correlation and challenge the proposed extent with 0.75× and 1.25× footprints. Seek the same template on a full adjacent cell, allowing a strong held-out match to stand alone. Source masks, atlas candidates and chart proposals come from the ordinary pipeline, so the held-out pixels are independent of this verification fit only.","",
           "The ridge logistic model uses these same features with one frozen probability threshold of 0.70. Its training data comprises 318 wall seeds and 292 positives from Watchpost 2, Observatory 02 and Riverside Mill 2. Each layout has equal weight. The six supplemental images use leave-one-layout-out predictions. Other panels use the model fit on all three training layouts. The rule and fitting recipe were fixed before measuring the two final six-frame panels. The native qualification control gives the same gate exact labels and is a diagnostic of that decision, not an oracle for final placement.","",
           "## Seed and chart qualification","","Seed precision and recall use the native audit criterion. Chart precision and recall grade whether an accepted chart contains any aligned seed. These denominators include only the wall seeds that this experiment can gate.","",
           "| Panel | Method | Seed precision | Seed recall | Chart precision | Chart recall |",
           "|---|---|---:|---:|---:|---:|"]
    for group in [*scores,"all"]:
        cases=[c for c in classification["cases"] if group=="all" or c["group"]==group]
        for name,key in (("Ungated",None),("Atlas rule","heuristic"),("Logistic","logistic")):
            lines.append(f"| {group} | {name} | {stats(cases,key)} |")
    lines.extend(["","## Decoded placements","","Strict / facade F1 grade all visible positions and facade-priority positions respectively. ID scores require the correct block at the same registered coordinate. Each pair uses the geometry-selected translation. A baseline or candidate with zero blocks is included.","",
                  "| Panel | Method | Strict | Facade | Strict + ID | Facade + ID | ID correct among correct positions | Correct / missed / extra | Recovered / lost / extras added / removed |",
                  "|---|---|---:|---:|---:|---:|---:|---|---|"])
    manifest=[]
    source_features=[]
    for group,data in scores.items():
        _,_,whole,cases=panel(group)
        for key,m in data["cohorts"]["all"].items():
            pos=m["geometry_strict"]
            ident=m["scores_strict"]
            f1=" | ".join(f"{m[k]['f1']*100:.2f}" for k in ("geometry_strict","geometry_facade","scores_strict","scores_facade"))
            diff=" / ".join(str(m["paired"][k]) for k in ("gained","lost","added","removed"))
            lines.append(f"| {group} | {key} | {f1} | {ident['correct']}/{pos['correct']} | {pos['correct']} / {pos['missed']} / {pos['extras']} | {diff} |")
        for c in cases:
            name=c["id"]
            source_features.append(dict(group=group,**read(OUT/group/"evidence"/(name+".json"))))
            row=dict(group=group,id=name,dataset=c["dataset"],split=c["split"],layout=c["layout"],outputs={})
            for key in data["cohorts"]["all"]:
                if key=="horizontal":
                    assembly=whole.parents[1]/"horizontal-cells/assembly" if group in ("holdout3","holdout4") else baseline(group,True)
                    charts=whole
                else:
                    dest=OUT/group if key=="verified" else OUT/group/key
                    assembly=dest/"assembly"
                    charts=dest/"charts"
                score=read(assembly/"connected-atlas/sky"/name/"score.json")
                if key=="horizontal":
                    row.update({k:score[k] for k in ("source_sha256","reference_sha256","surfaces_sha256","counts_sha256")})
                assert score["source_sha256"]==row["source_sha256"] and score["reference_sha256"]==row["reference_sha256"]
                packet=read(charts/"sky"/name/"charts.json")
                cells=[s for chart in packet["charts"] for s in chart["cells"]]
                row["outputs"][key]=dict(pack=score["pack_sha256"],charts=score["chart_sha256"],assigned=sum(not s.get("fallback",False) for s in cells),fallback=sum(bool(s.get("fallback")) for s in cells))
            manifest.append(row)
    save(REVIEW/"source-features.json.gz",source_features)
    save(REVIEW/"manifest.json.gz",dict(atlas=atlas().provenance,cases=manifest))
    lines.extend(["","## Chapel and Observatory","","On floating Chapel the weaker proposed seed failed the atlas rule. The second seed passed on the wrong froglight-side template, with held-out correlation 0.716, boundary margin 0.108 and neighbor correlation 0.660. Both seeds cover only a fraction of their recorded calcite faces. Both source gates kept the expansion. The native gate recovered the ordinary two-block chart, losing 116 correct placements and removing 1,431 extras. All three gates retained the two Observatory exports.","",
                  "The fixed 104 production views were unmeasured in this experiment. The carried-forward production trend is 91.27% strict and 93.42% facade (20,773 correct / 2,929 missed / 1,046 extra strict). These experimental builds use the floating branch's 86 frame conditions. The original panel contains official test views and was never used for model fitting. The initial six floating-sky conditions reuse some earlier layouts. Source and reference hashes, panel splits, pack hashes and assigned/fallback counts at chart input are in the manifest.","",
                  "From this checkout, run the following with the benchmark Python after producing the sibling audit inputs.","","'''powershell",
                  "python -m benchmark.floating_face_verifier","python -m benchmark.floating_face_verifier holdout3 holdout4","python -m benchmark.floating_face_model","python -m benchmark.floating_face_verifier_report","'''",""])
    lines.extend(["## Transfer and fixture coverage","","The Stable Yard high-right frame accounts for the complete holdout3 difference. The rule lost 45 net correct positions and removed 1,198 extras on this frame. The other five exports tied the baseline.","","![Measured seed decisions on source-image crops](seed-decisions.jpg)","","Across 86 conditions the comparison covers 35 distinct Visible Builds v3 frames out of 144, 23 Decorative Builds v1 frames out of 40 and 26 Visible Builds v2 frames. Dataset rows pool frame conditions with the same evaluator, including repeated source layouts under different backgrounds.","","| Dataset | Method | Strict | Facade | Strict + ID | Facade + ID |","|---|---|---:|---:|---:|---:|"])
    for dataset in ("all","visible-builds-v2","visible-builds-v3","decorative-builds-v1"):
        for arm in ("horizontal","verified","logistic","oracle"):
            rates=[]
            for kind in ("geometry_strict","geometry_facade","scores_strict","scores_facade"):
                sums=Counter()
                for data in scores.values():
                    if dataset in data["cohorts"]:
                        sums.update(data["cohorts"][dataset][arm][kind])
                correct,missed,extra=(sums[k] for k in ("correct","missed","extras"))
                rates.append(f"{200*correct/(2*correct+missed+extra):.2f}")
            lines.append("| "+" | ".join([dataset,arm,*rates])+" |")
    overlay()
    (REVIEW/"README.md").write_text("\n".join(lines).replace("'''",chr(96)*3),encoding="utf-8")


if __name__=="__main__":
    main()
