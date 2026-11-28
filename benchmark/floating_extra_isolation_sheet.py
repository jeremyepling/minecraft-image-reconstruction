"""Draw the chart substitution and its effect on visible, registered truth positions."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_bridges_report import read_arm
from benchmark.floating_extra_isolation import OUT, PREFIX, REVIEW, baseline, finder, handoffs, read

EXAMPLES = [
    ("sky", "validation-hillside-chapel-1-warm-low-left", "Chapel, low left"),
    ("holdout4", "train-riverside-mill-1-warm-high-right", "Riverside Mill, high right"),
    ("holdout4", "train-riverside-mill-1-warm-low-left", "Riverside Mill, low left"),
    ("holdout5", "train-amphitheater-01-orchid-low-left", "Amphitheater, low left"),
]


def main():
    width, height = 460, 297
    sheet = Image.new("RGB", (width * 4, 60 + (height + 54) * len(EXAMPLES)), "#111a23")
    pen = ImageDraw.Draw(sheet)
    title = ImageFont.truetype("arial.ttf", 22)
    small = ImageFont.truetype("arial.ttf", 18)
    for j, header in enumerate(("Source image", "Original expanded grid", "All proposed replacements",
                                "Positions after replacement")):
        pen.text((width*j + 7, 17), header, font=title, fill="white")
    for i, (group, name, label) in enumerate(EXAMPLES):
        c = next(c for c in handoffs(group)[3] if c["id"] == name)
        row = read(OUT / group / "rows" / (name + ".json"))
        roots = (baseline(group), finder(group) / "assembly")
        with Image.open(c["variants"]["sky"]["image"]) as im:
            source = im.convert("RGB")
        positions = [read_arm(root / PREFIX / name)[1] for root in roots]
        frame = Path(c["frame"])
        faces = read(frame / "labels.json")["faces"]
        with np.load(frame / "labels.npz") as data:
            owner = data["face_id"]
        overlay = np.array(source)
        overlay = (overlay * .55).astype(np.uint8)
        for present, color in (((True, True), (56, 211, 157)), ((True, False), (251, 117, 96)),
                               ((False, True), (109, 187, 255))):
            ids = [f["id"] for f in faces if tuple(tuple(f["xyz_world"]) in p for p in positions) == present]
            overlay[np.isin(owner, ids)] = color
        images = [source]
        for root in roots:
            with Image.open(root / PREFIX / name / "source-overlay.jpg") as im:
                images.append(im.convert("RGB"))
        images.append(Image.fromarray(overlay))
        y = 60 + i * (height + 54)
        for j, pic in enumerate(images):
            pic.thumbnail((width - 8, height), Image.Resampling.LANCZOS)
            sheet.paste(pic, (width*j + 4, y))
        captions = [label]
        for arm in ("baseline", "source-one"):
            score = row["score"][arm]
            s = score["geometry"]["strict"]
            captions.append(f"{s['correct']} correct, {s['extras']} extra, {score['scores']['strict']['correct']} exact IDs")
        d = row["paired"]["source-one"]
        captions.append(f"{d['lost']} lost (coral), {d['gained']} gained (blue)")
        for j, caption in enumerate(captions):
            pen.text((width*j + 6, y + height), caption, font=small, fill="white")
    sheet.save(REVIEW / "chart-substitutions.jpg", quality=91)


if __name__ == "__main__":
    main()
