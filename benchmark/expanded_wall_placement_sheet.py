"""Show which visible reference positions the frozen, registered exports recovered."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import SOURCE, WORKTREES


def main():
    examples = [("visible-builds-v2", "twinspire1-cool-high-right"), ("visible-builds-v3", "castle-01-ashlar-high-right"),
                ("decorative-builds-v1", "walled-garden-2-warm-low-left")]
    runs = [("Ordinary floating pipeline", "crop-ceiling-expanded/runs/native-crop-ceiling/source"),
            ("Sample the whole wall", "expanded-wall-rulers/runs/wall-rulers/source_resample"),
            ("Exact local grids, estimated directions", "expanded-wall-grids/runs/wall-grids/source-normal")]
    sheet = Image.new("RGB", (1440, 684), "#162130")
    pen = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("arial.ttf", 16)
    for i, label in enumerate(["Source image", *(name for name, _ in runs)]):
        pen.text((i*360+6, 7), label, font=font, fill="white")
    for row, (dataset, key) in enumerate(examples):
        frames = SOURCE / dataset / "data/test/frames"
        frame = next((f for f in sorted(frames.iterdir()) if key in f.name), None)
        if not frame:
            raise ValueError((dataset,key))
        rgb = Image.open(frame / "rgb.png").convert("RGB")
        labels = json.loads((frame / "labels.json").read_text())["faces"]
        targets = {tuple(f["xyz_world"]) for f in labels}
        with np.load(frame / "labels.npz") as arrays:
            owners = arrays["face_id"]
        panels = [(rgb, frame.name.replace("test-", ""))]
        for _, folder in runs:
            path = WORKTREES / folder / "assembly/connected-atlas/sky" / frame.name / "placements.json"
            placed = {tuple(xyz) for xyz, _ in json.loads(path.read_text())["aligned"]}
            correct = targets & placed
            ids = [f["id"] for f in labels if tuple(f["xyz_world"]) in correct]
            mask = Image.fromarray(np.isin(owners,ids).astype("uint8") * 195)
            panel = rgb.copy()
            panel.paste(Image.new("RGB", panel.size, "#28ff94"), mask=mask)
            panels.append((panel, f"{len(correct):,} recovered of {len(targets):,} visible positions"))
        for col, (image, caption) in enumerate(panels):
            image.thumbnail((355, 185), Image.Resampling.LANCZOS)
            x, y = col*360+3, 35+row*215
            sheet.paste(image, (x,y))
            pen.text((x, y+image.height+2), caption, fill="white", font=font)
    dest = Path(__file__).parent / "reviews/expanded-wall-connectivity-01/placement.jpg"
    sheet.save(dest, quality=91)


if __name__ == "__main__":
    main()
