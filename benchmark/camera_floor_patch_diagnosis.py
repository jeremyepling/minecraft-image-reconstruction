"""Reference-only audit of the actual native verification samples on five abstentions."""

from itertools import product

import benchmark
import numpy as np
from PIL import Image

from benchmark.camera_metrics import truth_camera
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, read, save
from engine.camera_atlas_floor import _atlas


def main():
    native = WORKTREES / "camera-native-floor-inverse"
    benchmark.__path__.append(str(native / "benchmark"))
    from benchmark.camera_native_floor import atlas_scan, correlation, coords, patches, sample_uv, templates
    atlas = _atlas("26.3")
    ids = [atlas.labels.index("light_gray_concrete")]
    grades = read(ROOT / "benchmark/reviews/camera-recovery-next-01/native-calibration.json.gz")
    rows = []
    for row in grades["cases"]:
        if row["accepted"]:
            continue
        key = row["id"]
        frame = SOURCE / "visible-builds-v3/data" / row["split"] / "frames" / key
        with Image.open(frame / "rgb.png") as image:
            samples = patches(image, bounds=((.04, .34), (.61, .84), (.89, .99), (.40, .54)))
        with np.load(frame / "labels.npz") as labels:
            face = labels["face_id"]
        exact = truth_camera(read(frame / "camera.json"))
        ground = 1 + read(frame / "frame.json")["ground_y"]
        exact[:, 3] += exact[:, 1] * ground
        inv = np.linalg.inv(exact[:, [0, 2, 3]])
        # The release atlas is periodic in whole blocks. Move the calculation near zero.
        start = inv @ samples[0][0][:, 0]
        inv[:2] -= np.floor(start[:2] / start[2])[:, None] * inv[2]
        photo = atlas_scan(inv, samples[:2], atlas, ids)
        tile = templates(atlas, ids, photo["n"])[0, photo["orientation"]]
        phase = np.array(photo["phase"])[:, None]
        # With the correct chart only, fit a nonnegative mixture of adjacent mip
        # levels on the first region. The other three supply a separate check.
        n = photo["n"]
        sizes = [s for s in (2 * n, n, n // 2) if 2 <= s <= 16]
        signals = []
        for screen, _ in samples:
            signals.append(np.stack([
                sample_uv(templates(atlas, ids, s)[0, photo["orientation"]],
                          coords(inv, screen, s) + phase * s / n, "linear") for s in sizes]))
        choices = []
        for weights in product(range(11), repeat=len(sizes)):
            if sum(weights) == 10:
                choices.append((correlation(samples[0][1], np.array(weights) @ signals[0]), weights))
        _, weights = max(choices)
        regions = []
        for (screen, target), levels in zip(samples, signals):
            x, y = np.floor(screen[:2]).astype(int)
            signal = sample_uv(tile, coords(inv, screen, photo["n"]) + phase, photo["mode"])
            occupied = face[y, x] != 0
            regions.append(dict(samples=len(x), visible_structure=int(occupied.sum()),
                                native_signal_sd=float(target.std()), exact_chart_correlation=correlation(target, signal),
                                west_fit_mip_mixture_correlation=correlation(target, np.array(weights) @ levels),
                                excluding_labeled_structure_correlation=correlation(target[~occupied], signal[~occupied])))
        rows.append(dict(id=key, source_sha256=row["source_sha256"], mip=photo["n"], filter=photo["mode"],
                         diagnostic_mip_blend=dict(sizes=sizes, weights_in_tenths=list(weights)), regions=regions))
    output = ROOT / "benchmark/reviews/camera-recovery-next-01/five-abstention-regions.json"
    save(output, rows)
    for r in rows:
        print(r["id"], [(z["visible_structure"], z["samples"], round(z["exact_chart_correlation"], 3)) for z in r["regions"]])


if __name__ == "__main__":
    main()
