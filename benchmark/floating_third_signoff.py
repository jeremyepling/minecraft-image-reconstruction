"""Build the third feedback session with native overlays supplied by each experiment."""
import json
from pathlib import Path

from benchmark.floating_grid_signoff import CASES, WORKTREES, make, read
from benchmark.floating_second_signoff import LOCATION, OLD_ROW, PART, REFIT, depth_options, option, read_arm

DEST = Path(__file__).parent/"reviews/floating-grid-round-04"
NOTES = [
    "The brick grid reaches five additional partial cells beside the sandstone. Four each isolate one visible brick face. The fifth crosses two faces. The exporter still selects another group. I would attach this wall before extending its grid further.",
    "Five projected cells mostly cover a gray cloud. All five place extra blocks. The pink depth patch contributes none of them. I would preserve partial roof cells and test whether the gray patch connects to the cloud outside the building.",
    "The far-left column already has 24 measured faces. The change attaches its grid to the selected build and uses the repeated log texture to position the four log blocks. Look at the export overlay to see what the local grid comparison cannot show.",
]


def build():
    rows = []
    for index, case in enumerate(CASES[:3]):
        kind,panel,name,wall,title,_ = case
        rel = Path("sky")/name/"charts.json"
        if index == 0:
            folders = [OLD_ROW/panel/"row-with-material", WORKTREES/"floating-occlusion-fragments/runs/occlusion-fragments"/panel]
            charts = [option(f,rel,wall) for f in folders]
            scores = [read_arm(f/"assembly"/LOCATION/name)[0] for f in folders]
        elif index == 1:
            record = next(r for r in read(REFIT/"conditions.json.gz") if r["panel"] == panel and r["id"] == name)
            current = depth_options(PART/panel/"source",record,rel,wall)[1]
            charts = [current,current]
            scores = [record["score"]["source-refit-with-small-steps"]]*2
        else:
            wall = "mask-1"
            records = read(WORKTREES/"floating-column-recovery/benchmark/reviews/floating-column-recovery-01/conditions.json.gz")
            record = next(r for r in records if r["panel"] == panel and r["id"] == name)
            current = option(PART/panel/"source",rel,wall)
            charts = [current,current]
            scores = [record["score"][key] for key in ("prior","repeated-texture-attachment")]
        row = make(("grid",panel,name,wall,title,NOTES[index]),dict(charts=charts,scores=scores))
        row["audit"] = ["twinspire.png","mill.jpg","viaduct.jpg"][index]
        rows.append(row)
    template = Path(__file__).with_name("floating_grid_signoff.html").read_text(encoding="utf-8")
    template = template.replace("floating-grid-signoff-02","floating-grid-round-04").replace("floating-grid-signoff-feedback.json","floating-grid-round-three-feedback.json")
    template = template.replace('<div class="controls">','<figure><figcaption>What changed at export or projection</figcaption><img id="audit" alt="Annotated experiment evidence"></figure><div class="controls">')
    template = template.replace('function images(){let c=cases[n],','function images(){$("audit").src=cases[n].audit;let c=cases[n],')
    template = template.replace('["context","a","b","ref","rgb"]','["audit","context","a","b","ref","rgb"]')
    (DEST/"review.html").write_text(template.replace("__PAYLOAD__",json.dumps(rows)),encoding="utf-8")


if __name__ == "__main__":
    build()

