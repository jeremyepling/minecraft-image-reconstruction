"""Build the comparison table and an offline source-image viewer from the scored ruler exports."""
import base64
from io import BytesIO
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_one_face_control import OUT,REVIEW,WORKTREES,inputs,read
from benchmark.relative_oracle_charts import UNIT
from engine.geometry import project


def jpeg(image):
    buf=BytesIO()
    image.save(buf,format="JPEG",quality=88)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")


def scorerows():
    scores={}
    for pattern in ("oracle-*.gz","precision-*.gz","shape-*.gz","stabilize-*.gz"):
        for path in sorted(REVIEW.glob(pattern)):
            for group,data in read(path).items():
                for case in data["rows"]:
                    dest=scores.setdefault((group,case["id"]),{})
                    dest.update(case["score"])
    return scores


def review(scores):
    choices=[("sky","chapel-1"),("original","castle-01-ashlar-high-right"),("supplemental","observatory-02-verdigris-high-right"),
             ("fresh","arcade2-cool-high-right"),("holdout","amphitheater-02-orchid-low-left"),("holdout2","shrine-02-ochre-high-right"),
             ("holdout3","factory-03-kiln-high-right"),("holdout4","riverside-mill-1-warm-high-right")]
    examples=[]
    panels=[]
    for group,substring in choices:
        if not any(g==group for g,n in scores):
            continue
        _,original,_,cases=inputs(group)
        case=next(c for c in cases if substring in c["id"])
        name=case["id"]
        diag=read(OUT/group/"audit"/(name+".json"))
        if diag["single"] is None:
            continue
        anchor=next(a for a in diag["anchors"] if a["chart"]==diag["single"])
        old=next(c for c in read(original/"sky"/name/"charts.json")["charts"] if c["id"]==diag["single"])
        corners=np.array(anchor["corners"])
        seeds=[project(np.array(old["H"]),np.array([c["u"],c["v"]])+UNIT) for c in old["cells"]]
        seeds.sort(key=lambda p:np.linalg.norm(p.mean(0)-corners.mean(0)))
        source=Image.open(case["variants"]["sky"]["image"]).convert("RGB")
        pane=source.copy()
        pen=ImageDraw.Draw(pane)
        for group_polys,color in ((seeds,"#ff9c40"),([corners],"#00d5ff")):
            for quad in group_polys:
                pts=list(map(tuple,quad))
                pen.line(pts+pts[:1],fill=color,width=3)
        center=corners.mean(0)
        radius=max(16,max(np.ptp(corners,axis=0))*1.25)
        box=tuple(map(int,(*np.floor(center-radius),*np.ceil(center+radius))))
        crop=pane.crop(box).resize((240,240),Image.Resampling.NEAREST)
        key={"holdout3":"new6","holdout4":"new7"}.get(group,group)
        offers=read(WORKTREES/"floating-face-proposals/runs/face-proposals"/key/"offers"/(name+".json.gz"))
        row=next((r for r in offers["charts"] if r["id"]==diag["single"]),{})
        found=None
        if row.get("picked") is not None:
            q=np.asarray(row["offers"][row["picked"]]["quad"])
            pts=list(map(tuple,q))
            pen.line(pts+pts[:1],fill="#66ed85",width=3)
            center=q.mean(0)
            radius=max(16,max(np.ptp(q,axis=0))*1.25)
            box=tuple(map(int,(*np.floor(center-radius),*np.ceil(center+radius))))
            found=jpeg(pane.crop(box).resize((240,240),Image.Resampling.NEAREST))
        scaled=pane.copy()
        scaled.thumbnail((540,305),Image.Resampling.LANCZOS)
        data=scores[group,name]
        stats={key:[s["geometry"]["strict"][k] for k in ("correct","missed","extras")]+
                   [s["scores"]["strict"]["correct"]] for key,s in data.items()}
        examples.append(dict(name=name,chart=diag["single"],group=group,full=jpeg(scaled),zoom=jpeg(crop),found=found,
                             stats=stats,before=anchor["original"],after=anchor["proposed"]))
        if len(panels)<3:
            thumb=crop.resize((165,165),Image.Resampling.LANCZOS)
            panels.append((name,thumb,stats))
    sheet=Image.new("RGB",(len(panels)*310,260),"#15202d")
    pen=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("arial.ttf",15)
    pen.text((7,5),"Orange: original seeds     Blue: complete reference face",fill="white",font=font)
    for i,(name,thumb,stats) in enumerate(panels):
        x=i*310+8
        sheet.paste(thumb,(x,30))
        pen.text((x,203),name.replace("validation-","").replace("test-","").replace("train-",""),font=font,fill="white")
        a,b=stats["horizontal"],stats["one-view"]
        pen.text((x,228),f"Correct: {a[0]} → {b[0]}    Extra: {a[2]} → {b[2]}",font=font,fill="white")
    sheet.save(REVIEW/"seed-comparison.jpg",quality=92)
    html="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One visible face · ruler review</title><style>
*{box-sizing:border-box}body{max-width:1030px;margin:1.5rem auto;padding:0 1rem;background:#15202d;color:#eef2f5;font:1rem/1.45 Arial,sans-serif}h1{font-size:1.55rem}button,select{font:inherit;padding:.4rem;background:white;color:#15202d;border:0;border-radius:4px}nav{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.images{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:1rem}img{max-width:100%;height:auto}.old{color:#ff9c40}.new{color:#00d5ff}table{border-collapse:collapse}td,th{padding:.35rem .6rem;text-align:right;border-bottom:1px solid #6c7b84}th:first-child,td:first-child{text-align:left}</style>
<h1>One visible face as a ruler</h1><nav><button id="prev" aria-label="Previous view">←</button><select id="menu" aria-label="Source image"></select><button id="next" aria-label="Next view">→</button></nav>
<p><strong class="old">Orange</strong> marks the original seeds. <strong class="new">Blue</strong> marks a complete reference face near those seeds. Green marks a replacement located by the source-only face finder, when available.</p>
<div class="images"><img id="full" alt="Source image with seed outlines"><img id="zoom" alt="Original seeds and nearby reference face"><img id="found" alt="Replacement located from source pixels"></div><p id="desc"></p>
<table><thead><tr><th>Placement method</th><th>Correct</th><th>Missed</th><th>Extra</th><th>Correct position and ID</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=PAYLOAD;const labels={horizontal:'Current floating candidate','one-view':'One reference face on the selected chart','one-chart':'One reference face on each eligible chart','one-chart-own-plane':'Each reference face and its visible surface','pixel-center':'One face per chart, corners rounded','two-pixel-center':'Two rounded faces with their known offset'};
const menu=document.getElementById('menu'); data.forEach(d=>menu.add(new Option(d.name,d.name)));
function show(i){i=(i+data.length)%data.length;menu.selectedIndex=i;let d=data[i];document.getElementById('full').src=d.full;document.getElementById('zoom').src=d.zoom;document.getElementById('desc').textContent=d.chart+' · '+d.before+' proposed cells before calibration, '+d.after+' after using one reference face. Scores below apply to the whole image.';let body=document.getElementById('rows');body.replaceChildren();for(let [k,title] of Object.entries(labels)){if(!d.stats[k])continue;const row=body.insertRow();[title,...d.stats[k]].forEach(v=>row.insertCell().textContent=v)}}
menu.onchange=()=>show(menu.selectedIndex);document.getElementById('prev').onclick=()=>show(menu.selectedIndex-1);document.getElementById('next').onclick=()=>show(menu.selectedIndex+1);show(0)</script></html>"""
    html=html.replace("horizontal:'Current floating candidate',","horizontal:'Current floating candidate','source-one':'Replacement from source pixels','square-challenge':'Replace a non-square seed','prior-perspective':'Rounded face with source-estimated perspective',")
    html=html.replace("document.getElementById('desc').textContent=", "const patch=document.getElementById('found');patch.hidden=!d.found;if(d.found)patch.src=d.found;document.getElementById('desc').textContent=")
    (REVIEW/"source-review.html").write_text(html.replace("PAYLOAD",json.dumps(examples)),encoding="utf-8")


if __name__=="__main__":
    review(scorerows())
