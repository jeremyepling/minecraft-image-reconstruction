"""Source-only veto for whole surface masks on the six frozen open-sky composites."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from engine.camera_lines import _runs
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


FIXTURES = SOURCE / "floating-build-grid-01/fixtures.json"
SURFACES = WORKTREES / "floating-build-local-grid/runs/floating-build-grid-01/sky"
CHARTS = WORKTREES / "surface-crop-first/runs/surface-crop-first-01/registered-seam/sky"
OUTPUT = ROOT / "runs/sky-mask-feedback-01"


def sky_support(image):
    """Compare each pixel to its row's border sky and plausible whitening by cloud."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    left, right = rgb[:, 0], rgb[:, -1]
    # Require independent agreement from two sides, and a blue, unsaturated-red sky.
    row_ok = ((np.abs(left - right).max(axis=1) <= 2) &
              (left[:, 2] >= 230) & (left[:, 2] - left[:, 0] >= 40) &
              (left[:, 1] - left[:, 0] >= 15))
    base = left[:, None, :]
    delta = rgb - base
    white_mix = delta[:, :, 0] / np.maximum(1, 255 - base[:, :, 0])
    expected_green = base[:, :, 1] + white_mix * (255 - base[:, :, 1])
    # A pale cloud must retain the sky's blue tint and similar mixing in R/G.
    compatible = (row_ok[:, None] & (delta[:, :, 0] >= -2) &
                  (np.abs(rgb[:, :, 1] - expected_green) <= 6) &
                  (np.abs(delta[:, :, 2]) <= 4) & (rgb[:, :, 2] - rgb[:, :, 0] >= 7))
    result = np.zeros_like(compatible)
    width = image.width
    for runs in _runs(compatible):
        if any(a == 0 or b == width for _, a, b in runs):
            for y, a, b in runs:
                result[y, a:b] = True
    return result


def predict(image, planes):
    sky = sky_support(image)
    labels = {p["id"]: f"S{i:02}" for i, p in
              enumerate(sorted(planes, key=lambda p: -p["support_area"]), 1)}
    measured = []
    for plane in planes:
        alpha, box = _native_alpha(plane["support"], image.size)
        x0, y0, x1, y1 = box
        pixels = np.asarray(alpha) != 0
        supported = int((sky[y0:y1, x0:x1] & pixels).sum())
        area = int(pixels.sum())
        measured.append(dict(id=plane["id"], label=labels[plane["id"]], box=box,
                             mask_pixels=area, sky_supported_pixels=supported,
                             removed=supported == area,
                             trim_mixed=0.99 * area <= supported < area))
    return measured, sky


def evaluate(case):
    ident = case["id"]
    image_path = Path(case["variants"]["sky"]["image"])
    surfaces_path = SURFACES / ident / "source-handoff/surfaces.json"
    charts_path = CHARTS / ident / "charts.json"
    planes = read(surfaces_path)["planes"]
    image = Image.open(image_path).convert("RGB")
    measured, sky = predict(image, planes)
    source_hash = digest(image_path)
    charts = read(charts_path)
    assert source_hash == case["variants"]["sky"]["source_sha256"] == charts["source_sha256"]
    assert digest(surfaces_path) == charts["surfaces_sha256"]
    labels = Path(case["frame"]) / "labels.npz"
    assert digest(labels) == case["reference_sha256"]["labels.npz"]
    with np.load(labels, allow_pickle=False) as ref:
        truth = ref["face_id"] > 0
    assert truth.shape == (image.height, image.width)
    bright = np.asarray(image).min(axis=2) >= 180
    before = np.zeros_like(truth)
    after = np.zeros_like(truth)
    trimmed_after = np.zeros_like(truth)
    removed = []
    trims = {}
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for row, plane in zip(measured, planes):
        alpha, (x0, y0, x1, y1) = _native_alpha(plane["support"], image.size)
        pixels = np.asarray(alpha) != 0
        region = np.s_[y0:y1, x0:x1]
        row["visible_pixels"] = int((pixels & truth[region]).sum())
        row["bright_visible_pixels"] = int((pixels & truth[region] & bright[region]).sum())
        before[region] |= pixels
        if row["removed"]:
            removed.append(row["id"])
            draw.rectangle((x0, y0, x1-1, y1-1), outline="#ff6c42", width=3)
            draw.text((x0, y0), row["label"], fill="black", stroke_width=2, stroke_fill="white")
        else:
            after[region] |= pixels
            # Separate follow-up to the whole-mask rule. Salvage the non-sky fringe
            # in mixed proposals with at least 99% source-only sky compatibility.
            if row["trim_mixed"]:
                retained = pixels & ~sky[region]
                rejected = pixels & sky[region]
                runs = np.asarray(row_runs(retained)) + [y0, x0, x0]
                trims[row["id"]] = dict(label=row["label"], rejected=int(rejected.sum()),
                                        rejected_visible=int((rejected & truth[region]).sum()),
                                        retained=int(retained.sum()), retained_visible=int((retained & truth[region]).sum()),
                                        retained_support=dict(size=list(image.size), runs=runs.tolist()))
                trimmed_after[region] |= retained
                draw.rectangle((x0, y0, x1-1, y1-1), outline="#ffda51", width=3)
                draw.text((x0, y0), row["label"], fill="black", stroke_width=2, stroke_fill="white")
            else:
                trimmed_after[region] |= pixels
    OUTPUT.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((960, 600))
    canvas.save(OUTPUT / (ident + ".jpg"), quality=90)
    chart_ids = [chart["id"] for chart in charts["charts"]]
    return dict(id=ident, split=case["split"], dataset=case["dataset"], source_sha256=source_hash,
                surfaces_sha256=digest(surfaces_path), charts_sha256=digest(charts_path),
                reference_sha256=digest(labels), chart_ids=chart_ids, removed_ids=removed,
                intersecting_chart_ids=sorted(set(chart_ids) & set(removed)), masks=measured,
                mixed_trim=trims, intersecting_trim_chart_ids=sorted(set(chart_ids) & trims.keys()),
                mixed_after_unique_px=int(trimmed_after.sum()),
                mixed_lost_visible_unique_px=int((before & ~trimmed_after & truth).sum()),
                mixed_removed_background_unique_px=int((before & ~trimmed_after & ~truth).sum()),
                before_background_px=int((before & ~truth).sum()),
                after_background_px=int((after & ~truth).sum()),
                mixed_after_background_px=int((trimmed_after & ~truth).sum()),
                before_unique_px=int(before.sum()), after_unique_px=int(after.sum()),
                lost_visible_unique_px=int((before & ~after & truth).sum()),
                removed_background_unique_px=int((before & ~after & ~truth).sum()))


def main():
    cases = read(FIXTURES)["cases"]
    frames = [evaluate(c) for c in cases]
    save(OUTPUT / "report.json", dict(fixtures_sha256=digest(FIXTURES), frames=frames))
    for f in frames:
        print(f["id"], len(f["masks"]), f["removed_ids"], f["intersecting_chart_ids"],
              f["removed_background_unique_px"], f["lost_visible_unique_px"])


if __name__ == "__main__":
    main()
