"""Show aligned visible positions lost and recovered by replacing the wall grid."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import read
from benchmark.floating_bridges_report import read_arm
from benchmark.floating_extra_budget import LOCATION, REVIEW, WORKTREES, paths


def render():
    cases = [
        ("sky", "validation-hillside-chapel-1-warm-low-left", "Hillside Chapel"),
        ("holdout4", "train-riverside-mill-1-warm-high-right", "Riverside Mill, high right"),
        ("holdout4", "train-riverside-mill-1-warm-low-left", "Riverside Mill, low left"),
        ("holdout5", "train-amphitheater-01-orchid-low-left", "Amphitheater, sparse wall"),
    ]
    sheet = Image.new("RGB", (900, 888), "#142031")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arial.ttf", 17)
    title = ImageFont.truetype("arialbd.ttf", 22)
    draw.text((15, 10), "Which correct placements moved?", font=title, fill="white")
    draw.text((15, 41), "Orange = lost correct position     Green = recovered position     Gray = other pixels",
              font=font, fill="white")
    for i, (panel, name, caption) in enumerate(cases):
        inputs, before, after = paths(panel)
        case = next(c for c in inputs[3] if c["id"] == name)
        frame = Path(case["frame"])
        faces = read(frame / "labels.json")["faces"]
        if panel == "holdout5":
            after = WORKTREES / "floating-extra-isolation/runs/extra-isolation/holdout5/sparse-mask-0/assembly"
        old_score, old = read_arm(before / LOCATION / name)
        new_score, new = read_arm(after / LOCATION / name)
        truth = {tuple(f["xyz_world"]) for f in faces}
        lost = (old.keys() - new.keys()) & truth
        gained = (new.keys() - old.keys()) & truth
        with np.load(frame / "labels.npz") as labels:
            ids = labels["face_id"]
        with Image.open(case["variants"]["sky"]["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        colored = np.asarray(Image.fromarray(rgb).convert("L").convert("RGB")).copy()
        for points, tint in ((lost, (255, 125, 27)), (gained, (45, 244, 130))):
            mask = np.isin(ids, [f["id"] for f in faces if tuple(f["xyz_world"]) in points])
            colored[mask] = .24 * rgb[mask] + .76 * np.asarray(tint)
        y = 75 + i * 202
        for j, pix in enumerate((rgb, colored)):
            pic = Image.fromarray(pix)
            pic.thumbnail((305, 184), Image.Resampling.LANCZOS)
            sheet.paste(pic, (10 + 311*j, y))
        draw.text((637, y+12), caption, font=font, fill="white")
        draw.text((637, y+47), f"{len(lost)} lost, {len(gained)} recovered", font=font, fill="white")
        old_extras = old_score["geometry"]["strict"]["extras"]
        new_extras = new_score["geometry"]["strict"]["extras"]
        draw.text((637, y+82), f"Extras  {old_extras:,} to {new_extras:,}", font=font, fill="white")
    sheet.save(REVIEW / "placement-changes.png")


if __name__ == "__main__":
    render()
