"""Fit a small confidence model to the native renderer's actual numerical options."""

import argparse
from collections import Counter
import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save


NATIVE = WORKTREES / "camera-native-floor-inverse/runs/camera-native-floor-inverse-01"
AUDIT = WORKTREES / "camera-native-floor-inverse/benchmark/reviews/camera-native-floor-inverse-01/native-cameras.json.gz"
CONTROLS = ROOT / "benchmark/reviews/camera-model-controls-01/independent-matched-cases.json.gz"
REVIEW = ROOT / "benchmark/reviews/camera-native-fit-confidence-01"
RUN = ROOT / "runs/camera-native-fit-confidence-01"
FEATURES = ["worst_atlas_patch", "median_atlas_patch", "patch_spread",
            "weakest_axis_repeat", "median_repeat", "repeat_minus_half_block", "two_band_consistency"]


def featurize(source, prop):
    scores = np.array(prop["native_correlation"])
    repeats = test_shifts(source, prop["P"])
    if repeats is None:
        return None
    return [float(scores.min()), float(np.median(scores)), float(scores.std()), *repeats]


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / max(1.e-8, np.linalg.norm(a) * np.linalg.norm(b)))


def source_signal(image):
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    center = np.median(gray[int(.79 * h)::4, ::4])
    good = cv2.erode((abs(gray - center) < 5).astype("uint8"), np.ones((9, 9), "uint8")).astype(bool)
    clipped = np.clip(gray, center - 5, center + 5)
    signal = clipped - cv2.GaussianBlur(clipped, (0, 0), 5)
    samples = []
    # The top band extends above the native atlas verifier. Integer tile shifts
    # are verified on source-source pairs without querying an atlas.
    for start, stop in [(.70, .79), (.84, .97)]:
        y, x = np.mgrid[int(h * start):int(h * stop):3, 12:w - 12:7]
        keep = good[y, x]
        samples.append((x[keep], y[keep]))
    return signal, good, samples


def test_shifts(source, camera):
    signal, good, samples = source
    H = np.asarray(camera)[:, [0, 2, 3]]
    inv = np.linalg.inv(H)
    bands = []
    for x, y in samples:
        if len(x) < 200:
            return None
        image = np.array([x + .5, y + .5, np.ones_like(x)])
        ground = inv @ image
        ground /= ground[2]
        correlations = []
        for dx, dz in [(1, 0), (0, 1), (2, 1), (1, 2), (.5, 0), (0, .5)]:
            shifted = H @ (ground + [[dx], [dz], [0]])
            u, v = (shifted[:2] / shifted[2] - .5).astype(np.float32)
            valid = cv2.remap(good.astype("uint8"), u, v, cv2.INTER_NEAREST).ravel().astype(bool)
            if valid.sum() < 200:
                return None
            moved = cv2.remap(signal, u, v, cv2.INTER_LINEAR).ravel()
            correlations.append(correlation(signal[y[valid], x[valid]], moved[valid]))
        bands.append(correlations)
    scores = np.array(bands)
    full, half = scores[:, :4], scores[:, 4:]
    return [float(full[:, :2].min()), float(np.median(full)), float(np.median(full) - half.max()),
            float(np.median(full, axis=1).min())]


def observations(split):
    cases = []
    for row in read(CONTROLS):
        if row["camera_usable"] or row["split"] != split:
            continue
        key = row["id"]
        path = SOURCE / row["dataset"] / "data" / split / "frames" / key / "rgb.png"
        assert digest(path) == row["source_sha256"]
        with Image.open(path) as image:
            source = source_signal(image)
        options = []
        for arm in ("boundary", "geocalib"):
            receipt = read(NATIVE / arm / "metric" / (key + ".json"))
            assert receipt["rgb_sha256"] == row["source_sha256"]
            for index, prop in enumerate(receipt["camera"]["evidence"].get("refined_proposals", [])):
                features = featurize(source, prop)
                if features is None:
                    continue
                options.append(dict(arm=arm, index=index, x=features))
        incumbent = read(NATIVE / "metric" / (key + ".json"))["camera"]["P"] is not None
        cases.append(dict(id=key, family=row["family"], split=split, rgb_sha256=row["source_sha256"],
                          incumbent=incumbent, options=options))
    return cases


def label_training(cases):
    refs = {r["id"]: r for r in read(AUDIT) if r["split"] == "train"}
    rows = []
    for case in cases:
        seen = set()
        for opt in case["options"]:
            fingerprint = tuple(np.round(opt["x"], 2))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            ref = refs[case["id"]]["arms"][opt["arm"]]["refined"][opt["index"]]["native"]
            rows.append(dict(id=case["id"], family=case["family"], x=opt["x"],
                             y=int(ref.get("p90_check_px", 1.e6) <= 2)))
    return rows


def fit(rows):
    counts = Counter(r["id"] for r in rows)
    weights = np.array([1 / counts[r["id"]] for r in rows])
    x = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    mean = np.average(x, axis=0, weights=weights)
    sd = np.maximum(.05, np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)))
    z = np.column_stack([np.ones(len(x)), (x - mean) / sd])
    coeff = np.zeros(z.shape[1])
    penalty = np.diag([0.] + [1.] * x.shape[1])
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(z @ coeff, -35, 35)))
        gradient = z.T @ ((p - y) * weights) + penalty @ coeff
        hessian = (z.T * (weights * p * (1 - p))) @ z + penalty
        step = np.linalg.solve(hessian, gradient)
        coeff -= step
        if np.linalg.norm(step) < 1.e-7:
            break
    return dict(mean=mean.tolist(), sd=sd.tolist(), coefficients=coeff.tolist())


def probability(model, features):
    z = (np.array(features) - model["mean"]) / model["sd"]
    return float(1 / (1 + np.exp(-np.clip(np.r_[1., z] @ model["coefficients"], -35, 35))))


def negative_scores(model):
    """Replay the exact phase edits from the frozen source challenger's negative suite."""
    receipts = read(AUDIT.parent / "negative-controls.json")
    folder = SOURCE / "visible-builds-v3/data/train/frames/train-pillars-02-marble-high-right"
    original = np.array(Image.open(folder / "rgb.png").convert("RGB"))
    h, w, _ = original.shape
    scores = {}
    for row in receipts:
        if row["expected_acceptance"]:
            continue
        options = row["evidence"].get("refined_proposals", [])
        if not options:
            scores[row["fixture"]] = []
            continue
        image = original.copy()
        if row["fixture"] == "split_phase":
            image[:, w // 2:] = np.roll(original, 37, axis=1)[:, w // 2:]
        elif row["fixture"] == "near_far_phase":
            image[int(.88 * h):] = np.roll(original, 31, axis=1)[int(.88 * h):]
        else:
            raise ValueError(f"Missing regenerated fixture: {row['fixture']}")
        source = source_signal(Image.fromarray(image))
        vectors = [featurize(source, option) for option in options]
        scores[row["fixture"]] = [probability(model, v) for v in vectors if v is not None]
    return scores


def train():
    cases = observations("train")
    rows = label_training(cases)
    oof = []
    for family in sorted({r["family"] for r in rows}):
        model = fit([r for r in rows if r["family"] != family])
        for r in rows:
            if r["family"] == family:
                oof.append(dict(r, probability=probability(model, r["x"])))
    trained = fit(rows)
    negatives = negative_scores(trained)
    bad = [r["probability"] for r in oof if not r["y"]]
    bad.extend(v for values in negatives.values() for v in values)
    cutoff = max(.90, .02 + max(bad))
    model = dict(trained, cutoff=cutoff, feature_names=FEATURES, training_candidates=len(rows),
                 family_folds=sorted({r["family"] for r in rows}))
    save(REVIEW / "calibration-negatives.json", negatives)
    save(REVIEW / "model.json", model)
    save(REVIEW / "leave-family-out.json.gz", oof)
    save(RUN / "train-source.json", cases)
    print("candidates", len(rows), "accurate", sum(r["y"] for r in rows), "cutoff", cutoff,
          "oof true accepted", sum(r["y"] and r["probability"] >= cutoff for r in oof), flush=True)


def predict(split):
    model = read(REVIEW / "model.json")
    cases = observations(split)
    for r in cases:
        opts = r["options"]
        for option in opts:
            option["confidence"] = probability(model, option["x"])
        ranked = sorted(opts, key=lambda o: -o["confidence"])
        best = ranked[0] if ranked else None
        r["selected"] = None if not best or r["incumbent"] or best["confidence"] < model["cutoff"] else best
    save(REVIEW / f"{split}-decisions.json.gz", cases)
    print(split, "new decisions", [(r["id"], r["selected"]) for r in cases if r["selected"]], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["fit", "train", "validation", "test"])
    step = p.parse_args().step
    train() if step == "fit" else predict(step)
