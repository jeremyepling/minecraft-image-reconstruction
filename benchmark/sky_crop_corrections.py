"""Reference-assisted iterative correction of existing native sky proposal masks."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import benchmark
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read
from benchmark.sky_crop_clicks import make_predictor, predict_clicks
from benchmark.sky_crop_models import BASELINE, MODELS, write_json
from engine.plane_consensus import row_runs

benchmark.__path__.append(str(WORKTREES / 'sky-crop-reference/benchmark'))
from benchmark.sky_crop_reference import best_target, intersection, native_mask, reference_raster

OUTPUT = ROOT / 'runs/sky-crop-corrections'
MAX_CLICKS = 20
PATIENCE = 5
THRESHOLD = .95


def overlap(mask, target):
    hit = int(np.count_nonzero(mask & target))
    return hit / int(np.count_nonzero(mask | target))


def deepest_pixel(error, points):
    # Exclude prior clicks and a small disk to permit exploration after a rejected edit.
    x0, y0, w, h = cv2.boundingRect(error.astype(np.uint8))
    if not w:
        return None
    local = np.pad(error[y0:y0+h, x0:x0+w].astype(np.uint8), 1)
    for x, y in points:
        cv2.circle(local, (x-x0+1, y-y0+1), 2, 0, -1)
    distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    _, maximum, _, (x, y) = cv2.minMaxLoc(distance)
    if maximum == 0:
        return None
    return (maximum, x+x0-1, y+y0-1)


def next_click(mask, target, points):
    options = []
    for label, error in [(1, target & ~mask), (0, mask & ~target)]:
        candidate = deepest_pixel(error, points)
        if candidate:
            radius, x, y = candidate
            options.append((radius, label, x, y))
    return max(options) if options else None


def diagnose(ref, idx, target_idx, x, y, label):
    if label:
        return 'missing target pixels'
    if idx:
        owner, target = ref.targets[idx-1], ref.targets[target_idx-1]
        if owner['normal_axis'] != target['normal_axis']:
            return 'neighboring cube, different face direction'
        if owner['plane_offset'] != target['plane_offset']:
            return 'neighboring cube, parallel plane at a different offset'
        return 'neighboring cube, disconnected on same plane'
    return 'non-full-cube geometry' if ref.non_cube[y,x] else 'background'


def encode(mask):
    return dict(size=list(mask.shape[::-1]), runs=row_runs(mask))


def correct(predictor, ref, mask, target_idx, record):
    target = ref.labels == target_idx
    retained = encode(mask)
    snapshots = [retained]
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    box = [x, y, x+w-1, y+h-1]
    points, labels = [], []
    best = record['baseline_iou'] = overlap(mask, target)
    stalled = 0
    record['stop'] = 'click budget'
    for step in range(1, MAX_CLICKS+1):
        if best >= THRESHOLD:
            record['stop'] = 'reached .95 IoU'
            break
        click = next_click(mask, target, points)
        if not click:
            record['stop'] = 'no untried error point'
            break
        radius, label, x, y = click
        owner = int(ref.labels[y,x])
        diagnosis = diagnose(ref, owner, target_idx, x, y, label)
        points.append([x,y])
        labels.append(label)
        masks, predicted = predict_clicks(predictor, box, points, labels)
        scores = [overlap(candidate, target) for candidate in masks]
        chosen = int(np.argmax(scores))
        proposed = scores[chosen]
        accepted = proposed > best
        record['steps'].append(dict(round=step, x=x, y=y, label=label, diagnosis=diagnosis,
                                    error_pixels=int(np.count_nonzero((target & ~mask) if label else (mask & ~target))),
                                    click_owner=ref.targets[owner-1]['id'] if owner else None,
                                    click_radius=radius, before_iou=best, proposal_iou=proposed,
                                    proposed_ious=scores, model_ious=predicted.tolist(), chosen=chosen,
                                    accepted=accepted, after_iou=max(best,proposed)))
        if accepted:
            mask, best = masks[chosen], proposed
            retained = encode(mask)
            stalled = 0
        else:
            stalled += 1
        snapshots.append(retained)
        if stalled == PATIENCE:
            record['stop'] = 'five consecutive rejected attempts'
            break
    if best >= THRESHOLD:
        record['stop'] = 'reached .95 IoU'
    record['final_iou'] = best
    return snapshots


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictor = make_predictor()
    fixture = SOURCE / 'floating-build-grid-01/fixtures.json'
    rounds = [{} for _ in range(MAX_CLICKS+1)]
    trace = dict(checkpoint_sha256=MODELS['sam-b'][2], fixtures_sha256=digest(fixture), frames={}, inputs={})
    for case in read(fixture)['cases']:
        frame = case['id']
        path = Path(case['variants']['sky']['image'])
        ref_path = Path(case['frame'])
        handoff = BASELINE / frame / 'source-handoff/surfaces.json'
        source_hash = digest(path)
        ref_hash = {name:digest(ref_path/name) for name in case['reference_sha256']}
        assert source_hash == case['variants']['sky']['source_sha256'] and ref_hash == case['reference_sha256']
        trace['inputs'][frame] = dict(source_sha256=source_hash, reference_sha256=ref_hash, surfaces_sha256=digest(handoff))
        image = np.asarray(Image.open(path).convert('RGB'))
        ref = reference_raster(ref_path)
        predictor.set_image(image)
        rows = trace['frames'][frame] = {}
        for packet in rounds:
            packet[frame] = {}
        planes = sorted(read(handoff)['planes'], key=lambda p:-p['support_area'])
        for index,plane in enumerate(planes,1):
            ident = plane['id']
            mask = native_mask(plane['support'], image.shape[1::-1])
            target_idx = best_target(ref.areas, intersection(ref, mask), int(mask.sum()))
            record = rows[ident] = dict(label=f'S{index:02}', target=ref.targets[target_idx-1]['id'] if target_idx else None, steps=[])
            if target_idx:
                snapshots = correct(predictor, ref, mask, target_idx, record)
            else:
                record.update(baseline_iou=0, final_iou=0, stop='no overlapping cube target')
                snapshots = [encode(mask)]
            for number,packet in enumerate(rounds):
                packet[frame][ident] = snapshots[min(number,len(snapshots)-1)]
        print(frame, Counter(r['stop'] for r in rows.values()), flush=True)
    rows = [r for f in trace['frames'].values() for r in f.values()]
    assert len(rows)==193 and sum(bool(r['target']) for r in rows)==188
    actual_rounds = max(len(r['steps']) for r in rows)
    for number,packet in enumerate(rounds[:actual_rounds+1]):
        write_json(OUTPUT / f'round-{number:02}.json', dict(frames=packet))
    write_json(OUTPUT/'final.json', dict(frames=rounds[-1]))
    write_json(OUTPUT/'trace.json', trace)


if __name__ == '__main__':
    main()
