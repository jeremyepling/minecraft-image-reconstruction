"""Evaluate the frozen confidence gate on additional synthetic phase discontinuities."""

import benchmark
import numpy as np
from PIL import Image

from benchmark.camera_fit_confidence import NATIVE, REVIEW, SOURCE, featurize, probability, source_signal
from benchmark.expanded_baseline import digest, read, save


def main():
    source_root = NATIVE.parents[1]
    benchmark.__path__.append(str(source_root / "benchmark"))
    from benchmark.camera_native_floor_search import solve
    model = read(REVIEW / "model.json")
    key = "train-viaduct-02-riverstone-high-right"
    file = SOURCE / "visible-builds-v3/data/train/frames" / key / "rgb.png"
    original = np.array(Image.open(file).convert("RGB"))
    h, w, _ = original.shape
    fixtures = {"unmodified": original}
    for name, offset, region in [
        ("lateral_phase", 59, np.indices((h, w))[1] > w * .47),
        ("band_phase", -43, np.indices((h, w))[0] > h * .90),
        ("alternating_bands", 31, (np.indices((h, w))[0] // 49) % 2 == 0),
    ]:
        adjusted = original.copy()
        adjusted[region] = np.roll(original, offset, axis=1)[region]
        fixtures[name] = adjusted
    results = []
    for name, rgb in fixtures.items():
        image = Image.fromarray(rgb)
        native = solve(image, None)
        source = source_signal(image)
        chances = []
        for candidate in native["evidence"].get("refined_proposals", []):
            x = featurize(source, candidate)
            if x is not None:
                chances.append(probability(model, x))
        results.append(dict(fixture=name, boundary=native["P"] is not None,
                            model_score=max(chances, default=None),
                            added=bool(native["P"] is None and chances and max(chances) >= model["cutoff"])))
    save(REVIEW / "additional-controls.json", dict(source_sha256=digest(file), results=results))
    print(results)


if __name__ == "__main__":
    main()
