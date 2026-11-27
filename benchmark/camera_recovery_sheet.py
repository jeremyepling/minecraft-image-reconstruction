"""Show the photographed pavement on a model-seeded Castle recovery and an abstention."""

import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, read


def floor_patch(image, bounds, projection):
    x0, y0, x1, y1 = bounds
    gray = np.asarray(image.crop(bounds), dtype=float).mean(2)
    low, high = np.percentile(gray, [1, 99])
    amplified = np.clip(255 * (gray - low) / max(high - low, 1), 0, 255).astype("uint8")
    pixels = np.repeat(amplified[:, :, None], 3, axis=2)
    if projection is not None:
        ground = np.linalg.inv(np.asarray(projection)[:, [0, 2, 3]])
        y, x = np.mgrid[y0:y1, x0:x1]
        homogeneous = ground @ np.stack((x + .5, y + .5, np.ones_like(x))).reshape(3, -1)
        uv = (homogeneous[:2] / homogeneous[2]).reshape(2, y1 - y0, x1 - x0)
        unit_cells = np.floor(uv)
        edge = np.zeros(gray.shape, dtype=bool)
        edge[:, 1:] |= np.any(unit_cells[:, :, 1:] != unit_cells[:, :, :-1], axis=0)
        edge[1:, :] |= np.any(unit_cells[:, 1:] != unit_cells[:, :-1], axis=0)
        pixels[edge] = [255, 182, 42]
    return Image.fromarray(pixels).resize((684, 120), Image.Resampling.NEAREST)


def main():
    run = WORKTREES / "camera-native-floor-inverse/runs/camera-native-floor-inverse-01/metric"
    output = ROOT / "benchmark/reviews/camera-recovery-next-01"
    sheet = Image.new("RGB", (1440, 690), "#152224")
    text = ImageDraw.Draw(sheet)
    for column, (layout, title) in enumerate([
            (2, "Castle 02, limestone high right. Camera recovered."),
            (3, "Castle 03, limestone high right. Camera unresolved.")]):
        key = f"test-castle-0{layout}-limestone-high-right"
        frame = SOURCE / "visible-builds-v3/data/test/frames" / key
        image = Image.open(frame / "rgb.png").convert("RGB")
        projection = read(run / (key + ".json"))["camera"]["P"]
        bounds = (570, 813, 912, 873)
        patch = floor_patch(image, bounds, projection)
        ImageDraw.Draw(image).rectangle(bounds, outline="#ffbb32", width=4)
        x = column * 720
        text.text((x + 12, 9), title, font_size=21, fill="white")
        sheet.paste(image.resize((720, 450), Image.Resampling.LANCZOS), (x, 42))
        sheet.paste(patch, (x + 18, 516))
    text.text((18, 649), "Outlined source patch enlarged 2x with contrast enhanced. Gold lines mark the recovered one-block pavement grid.",
              fill="white", font_size=20)
    sheet.save(output / "castle-source-floor.jpg", quality=92)


if __name__ == "__main__":
    main()
