"""Create the feedback round's annotatable source and reference-assisted comparisons."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.floating_extra_feedback import grid, picture, read_arm, paths, _native_alpha, LOCATION, WORKTREES, read
from benchmark.floating_reviewed_rulers import OUT

ISOLATION = WORKTREES/"floating-extra-isolation/runs/extra-isolation"
SCALE = WORKTREES/"floating-feedback-scale/runs/feedback-scale"
SUPPORT = WORKTREES/"floating-feedback-support/runs/feedback-support"
QUESTIONS = {
    "scale": "Which face or repeating motif would let you reject the wrong block size? Does the grid agree after a material change?",
    "support": "Is there a visible reason to stop the grid here, or can the larger wall be supported by neighboring blocks?",
    "origin": "The alternative gets a reference-assisted origin shift with the same pitch. Where does alignment improve, and where does it drift?",
    "boundary": "Only the largest recorded surface is retained in B. Which visible areas should come back as separate surfaces?",
}
LABELS = {"scale":"Atlas scale", "support":"Smooth wall fallback", "origin":"Reference origin", "boundary":"Reference boundary"}


def example(arm, panel, name, wall, title):
    (src,_,_,cases),_,_ = paths(panel)
    case = next(c for c in cases if c["id"]==name)
    a = ISOLATION/panel/"large-contraction"
    if arm=="scale":
        # Both selected cases have no contraction substitutions, so union == scale alone.
        b = SCALE/panel
    elif arm=="support":
        b = SUPPORT/panel/"sparse"
    else:
        a = Path("runs/feedback-boundary")/panel/"resample"
        b = a.parent/arm
    with Image.open(case["variants"]["sky"]["image"]) as im:
        source = im.convert("RGB")
    plane = next(p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
    alpha,(x,y,r,bottom) = _native_alpha(plane["support"],source.size)
    box = (max(0,x-10),max(0,y-10),min(source.width,r+10),min(source.height,bottom+10))
    mask = Image.new("L",source.size)
    mask.paste(alpha,(x,y))
    dim = Image.blend(source,Image.new("RGB",source.size),.72)
    context = source.copy()
    ImageDraw.Draw(context).rectangle(box,outline="gold",width=4)
    panels = [("Source crop", source), ("Detected surface · surroundings dimmed",Image.composite(source,dim,mask))]
    stats = []
    for label,folder in (("A · Existing grid",a),("B · "+LABELS[arm],b)):
        chart = next(c for c in read(folder/"charts/sky"/name/"charts.json")["charts"] if c["id"]==wall)
        panels.append((label,grid(source,chart)))
        s,_ = read_arm(folder/"assembly"/LOCATION/name)
        g = s["geometry"]["strict"]
        stats.append([g["correct"],g["extras"],s["scores"]["strict"]["correct"],round(100*g["f1"],2)])
    with np.load(Path(case["frame"])/"labels.npz") as arrays:
        ids=arrays["face_id"]
    edges = np.zeros(ids.shape,bool)
    edges[1:] |= ids[1:]!=ids[:-1]
    edges[:,1:] |= ids[:,1:]!=ids[:,:-1]
    native = np.array(source)
    native[edges & (ids!=0)] = (255,210,0)
    panels.append(("Reference · visible face boundaries",Image.fromarray(native)))
    context.thumbnail((760,425))
    return dict(id=name+"-"+arm,title=LABELS[arm]+" · "+title,wall=wall,question=QUESTIONS[arm],context=picture(context),
                panels=[(label,picture(im.crop(box))) for label,im in panels],stats=stats)


def main():
    rows = [
        ("scale","holdout2","train-shrine-02-ochre-low-left","mask-0-wall-0","Shrine"),
        ("scale","original","test-twinspire2-earth-high-right","mask-2","Twinspire, earth palette"),
        ("support","holdout5","train-amphitheater-01-orchid-low-left","mask-0","Amphitheater 01"),
        ("support","holdout","train-amphitheater-02-orchid-high-right","mask-1-wall-0","Amphitheater 02"),
        ("support","supplemental","train-observatory-02-verdigris-low-left","mask-0","Observatory"),
        ("origin","sky","validation-hillside-chapel-1-warm-low-left","mask-0","Hillside Chapel"),
        ("origin","holdout4","train-riverside-mill-1-warm-high-right","mask-0","Riverside high right"),
        ("boundary","holdout4","train-riverside-mill-1-warm-low-left","mask-0","Riverside low left"),
    ]
    template = Path(__file__).with_name("floating_extra_review_template.html").read_text(encoding="utf-8")
    template = template.replace("floating-wall-review-v1","floating-feedback-experiments-01").replace(
        "floating-wall-feedback.json","floating-experiment-feedback.json")
    (OUT/"experiments.html").write_text(template.replace("__CASES__",json.dumps([example(*r) for r in rows])),encoding="utf-8")


if __name__=="__main__":
    main()
