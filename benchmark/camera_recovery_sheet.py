"""Show the photographed pavement on a model-seeded Castle recovery and an abstention."""

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_recorded_examples import COLORS, example
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, WORKTREES, frames, read


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
            (2, "Castle 02 limestone high right. GeoCalib-seeded experiment."),
            (3, "Castle 03 limestone high right. Both source methods abstained.")]):
        key = f"test-castle-0{layout}-limestone-high-right"
        frame = SOURCE / "visible-builds-v3/data/test/frames" / key
        image = Image.open(frame / "rgb.png").convert("RGB")
        projection = read(run / (key + ".json"))["camera"]["P"]
        if layout == 2:
            assert read(run / (key + ".json"))["arm"] == "geocalib" and projection is not None
        else:
            assert projection is None
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
    datasets = {e["id"]: e for e in frames(SOURCE, COHORTS)}
    scored = Image.new("RGB", (1160, 875), "#152224")
    paint = ImageDraw.Draw(scored)
    paint.text((10, 6), "Model-free boundary cameras and the resulting exported builds", font_size=24, fill="white")
    for n, (label, color) in enumerate(COLORS.items()):
        x = 12 + n * 287
        paint.rectangle((x, 42, x + 12, 54), fill=color)
        paint.text((x + 18, 37), label, font_size=15, fill="white")
    for index, (key, label) in enumerate([
        ("test-castle-01-limestone-high-right", "Castle 01 limestone high right"),
        ("validation-harbor-03-seafoam-high-right", "Harbor 03 seafoam high right"),
    ]):
        item = read(run.parent / "scores" / (key + ".json"))
        assert read(run / (key + ".json"))["arm"] == "boundary"
        photo, overlay, _ = example(datasets[key], run.parent / "builds", expected=item)
        y = 99 + index * 404
        paint.text((12, y - 29), label, font_size=18, fill="white")
        paint.text((590, y - 29), "Decoded export after whole-build alignment", font_size=18, fill="white")
        for x, view in [(12, photo), (590, overlay)]:
            scored.paste(view.resize((558, 349), Image.Resampling.LANCZOS), (x, y))
        s = item["geometry"]["strict"]
        paint.text((12, y + 351), "Source image", font_size=16, fill="white")
        paint.text((590, y + 351), f'{s["correct"]:,} correct, {s["missed"]:,} missed, {s["extras"]:,} extra positions.',
                   font_size=16, fill="white")
    scored.save(output / "decoded-castle-harbor.jpg", quality=91)


if __name__ == "__main__":
    main()
