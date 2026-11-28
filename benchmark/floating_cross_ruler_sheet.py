"""Source rulers, sampled wall cells, and correctly placed visible blocks for changed exports."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from benchmark.floating_cross_ruler import ARCHIVE, OUT, PREFIX, REVIEW, handoffs, read
from benchmark.floating_bridges_report import read_arm
from benchmark.relative_oracle_charts import UNIT
from engine.geometry import project


def evidence_sheet(rows):
    font = ImageFont.truetype("arial.ttf", 15)
    bold = ImageFont.truetype("arialbd.ttf", 17)
    width, height = 460, 288
    changed = [r for r in rows if any(d["take"] for d in r["decisions"])]
    sheet = Image.new("RGB", (width*3, 64+(height+56)*len(changed)), "#122134")
    draw = ImageDraw.Draw(sheet)
    for i, label in enumerate(("Source faces used as rulers", "Large-contraction control", "Cross-surface ruler")):
        draw.text((width*i+9,7),label,font=bold,fill="white")
    draw.text((9,37),"Cyan outlines mark atlas matches. Yellow lines show the affected chart. Green pixels mark correctly placed blocks after whole-build alignment.",font=font,fill="white")
    for index, row in enumerate(changed):
        group,name = row["group"],row["id"]
        case = next(c for c in handoffs(group)[3] if c["id"]==name)
        labels = read(Path(case["frame"])/"labels.json")["faces"]
        with np.load(Path(case["frame"])/"labels.npz") as fields:
            owners = fields["face_id"]
        chart_id = next(d["id"] for d in row["decisions"] if d["take"])
        source = Image.open(row["image"]).convert("RGB")
        rel = Path("sky")/name/"charts.json"
        offset = 64+index*(height+56)
        for i in range(3):
            pic = source.copy()
            if i==0:
                paint = ImageDraw.Draw(pic)
                for anchor in row["anchors"]:
                    vertices = [tuple(p) for p in anchor["quad"]]
                    paint.line(vertices+[vertices[0]], fill="#00efff", width=6)
                caption = ", ".join(a["atlas"] for a in row["anchors"])
            else:
                base = ARCHIVE/group/"large-contraction" if i==1 else OUT/group
                chart = next(c for c in read(base/"charts"/rel)["charts"] if c["id"]==chart_id)
                _, blocks = read_arm(base/"assembly"/PREFIX/name)
                correct = [f["id"] for f in labels if tuple(f["xyz_world"]) in blocks]
                mask = Image.fromarray((np.isin(owners,correct)*160).astype("uint8"))
                pic.paste(Image.new("RGB",pic.size,"#00d764"),mask=mask)
                paint = ImageDraw.Draw(pic)
                for c in chart["cells"]:
                    corners = project(np.array(chart["H"]), UNIT+[c["u"],c["v"]])
                    vertices = [tuple(p) for p in corners]
                    paint.line(vertices+[vertices[0]],fill="#ffc35c",width=1 if i==1 else 2)
                stats = row["score"]["control" if i==1 else "candidate"]["geometry"]["strict"]
                caption = f"{len(chart['cells'])} chart cells   |   {stats['correct']} correct   |   {stats['extras']} extra"
            pic.thumbnail((width,height))
            sheet.paste(pic,(i*width,offset))
            draw.text((i*width+8,offset+height+1),name if i==0 else chart_id,font=bold,fill="white")
            draw.text((i*width+8,offset+height+26),caption,font=font,fill="#bde6ff")
    sheet.save(REVIEW/"source-and-export.jpg",quality=92)


if __name__=="__main__":
    evidence_sheet(read(REVIEW/"conditions.json.gz"))

