"""Grade where an extra Surface Finder pass can act on the existing mask and cut inventory."""
import argparse
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import PROTOCOL, measure
from benchmark.normal_round import groups
from benchmark.single_wall_sheet import REVIEWS, checked, load_gzip
from benchmark.surface_reference import ARCHIVE, recorded
from engine.single_wall_gate import partition
from runtime.mode import image_only


SOURCE = REVIEWS / 'single-wall-depth-01'
MOBILE = REVIEWS / 'mobile-regions-01'
MISSES = REVIEWS / 'current-misses-01'
METHOD = 'single-v2'


def load_table(root, name, manifest):
    path = root / name
    checked(path, manifest['artifacts'][name])
    return read(path)


def candidate_masks(sha, mobile):
    stem = 'proposals/' + sha
    supplied = load_gzip(MOBILE, stem + '.json.gz', mobile['artifacts'])
    path = MOBILE / (stem + '.npz')
    checked(path, mobile['artifacts'][stem + '.npz'])
    with np.load(path, allow_pickle=False) as field:
        masks = np.unpackbits(field['packed'], count=int(field['shape'][2]), axis=2).astype(bool)
    if len(masks) != len(supplied['candidates']):
        raise ValueError('Saved MobileSAM masks and metadata disagree')
    areas = np.array([p['area'] for p in supplied['candidates']])
    return masks, areas, supplied['selected']


def ceiling_by_target(key, reference, missed, masks, areas, selected, inventory, parents):
    stages = {'raw':range(len(masks)), 'selected':selected,
              'offered':sorted({c['proposal'] for c in inventory}),
              'intact':sorted({c['proposal'] for c in inventory if c['parent'] in parents})}
    results = []
    for rid in missed['missed']:
        target = reference['id_map'] == rid
        intersections = masks[:, target].sum(axis=1)
        ious = intersections / (areas + target.sum() - intersections)
        best = {k:max((float(ious[i]) for i in indices), default=0.) for k, indices in stages.items()}
        results.append(dict(frame=key, reference=rid, orientation=missed['missed_orientation'][str(rid)],
                            **best))
    return results


def contrast_at_boundary(rgb, candidate):
    # Source RGB was averaged over each 4x4 square to align with the recorded sample pairs.
    _, inner, outer, _ = np.asarray(candidate['samples']).reshape(-1, 4, 2).transpose(1, 0, 2)
    difference = np.abs(rgb[inner[:, 0], inner[:, 1]] - rgb[outer[:, 0], outer[:, 1]])
    return float(np.median(difference.mean(axis=1)))


def inspect_frame(case, row, missed, source, mobile):
    key, sha = row['key'], row['sha256']
    checked(case.path('image'), sha)
    reference = recorded(case)[1]
    evidence = load_gzip(SOURCE, row['boundary_evidence'], source['artifacts'])
    baseline = load_gzip(SOURCE, row['predictions'], source['artifacts'])[METHOD]
    parent_indices = {p['id']:i for i, p in enumerate(baseline)}
    _, pairs = measure(baseline, reference, use_mask=True)
    found = {rid for _, rid in pairs}
    eligible = {rid for rid, face in reference['patches'].items() if face['eligible']}
    if found != set(missed['matched_ids']) or eligible - found != set(missed['missed']):
        raise ValueError(f'Published surface IDs disagree in {key}')
    masks, areas, selected = candidate_masks(sha, mobile)
    targets = ceiling_by_target(key, reference, missed, masks, areas, selected, evidence['inventory'], parent_indices)
    with Image.open(case.path('image')) as opened:
        image = opened.convert('RGB')
    rgb = np.asarray(image.resize(reference['id_map'].shape[::-1], Image.Resampling.BOX), dtype=float)
    boundaries, cuts = [], []
    for candidate, truth in zip(evidence['inventory'], evidence['reference_diagnostics'], strict=True):
        parent, proposal = candidate['parent'], candidate['proposal']
        if (parent, proposal) != (truth['parent'], truth['proposal']):
            raise ValueError(f'Boundary reference and candidate disagree in {key}')
        identifiers = dict(frame=key, split=row['cohort'], parent=parent, proposal=proposal, truth=truth['kind'])
        boundaries.append(dict(**identifiers, material_seam=truth['material_seam'],
                               contrast=contrast_at_boundary(rgb, candidate)))
        if parent not in parent_indices:
            continue
        index = parent_indices[parent]
        pieces = partition(image, baseline[index], proposal, masks[proposal])
        if not {'inside','remainder'} <= {p['single_wall_side'] for p in pieces}:
            continue
        trial = baseline[:index] + pieces + baseline[index + 1:]
        _, matched = measure(trial, reference, use_mask=True)
        result = {rid for _, rid in matched}
        cuts.append(dict(**identifiers, gains=sorted(result - found), losses=sorted(found - result)))
    return targets, boundaries, cuts


def contrast_score(samples, threshold):
    steps = np.array([s['contrast'] for s in samples if s['truth'] == 'parallel step'])
    flat = np.array([s['contrast'] for s in samples if s['truth'] == 'flush wall'])
    if not len(steps) or not len(flat):
        raise ValueError('The contrast diagnostic requires both boundary types')
    auc = (steps[:, None] > flat).mean() + .5 * (steps[:, None] == flat).mean()
    return dict(real=len(steps), flat=len(flat), real_passed=int((steps >= threshold).sum()),
                flat_passed=int((flat >= threshold).sum()), median_real=float(np.median(steps)),
                median_flat=float(np.median(flat)), auc=float(auc))


def contrast_summary(boundaries):
    development = [b for b in boundaries if b['split'] == 'development']
    regression = [b for b in boundaries if b['split'] == 'previously_reviewed_regression']
    thresholds = sorted({b['contrast'] for b in development} | {256.})
    def priority(threshold):
        s = contrast_score(development, threshold)
        balanced = .5 * (s['real_passed']/s['real'] + 1 - s['flat_passed']/s['flat'])
        return balanced, threshold
    threshold = max(thresholds, key=priority)
    return dict(threshold=threshold, development=contrast_score(development, threshold),
                regression=contrast_score(regression, threshold), all=contrast_score(boundaries, threshold))


def summarize(targets, boundaries, cuts):
    def gained(rows):
        return sorted({(r['frame'], rid) for r in rows for rid in r['gains']})
    return dict(misses=len(targets), sam={k:sum(r[k] >= PROTOCOL['iou'] for r in targets)
                                         for k in ('raw','selected','offered','intact')},
                raw_below_quarter=sum(r['raw'] < .25 for r in targets),
                raw_below_quarter_by_orientation=dict(Counter(r['orientation'] for r in targets if r['raw'] < .25)),
                raw_between_40_and_50_percent=sum(.4 <= r['raw'] < .5 for r in targets),
                inventory=len(boundaries), eligible_individual_cuts=len(cuts), individual_gain_ids=gained(cuts),
                isolated_gain_no_loss=gained(r for r in cuts if not r['losses']),
                cut_outcomes=dict(gain_only=sum(bool(r['gains']) and not r['losses'] for r in cuts),
                                  loss_only=sum(bool(r['losses']) and not r['gains'] for r in cuts),
                                  gained_and_lost=sum(bool(r['gains']) and bool(r['losses']) for r in cuts),
                                  no_change=sum(not r['gains'] and not r['losses'] for r in cuts)),
                contrast=contrast_summary(boundaries))


@image_only()
def run(out):
    source, mobile, misses = [read(p/'manifest.json') for p in (SOURCE, MOBILE, MISSES)]
    rows = load_table(SOURCE, 'cases.json', source)
    missed_rows = load_table(MISSES, 'frames.json', misses)
    sam_rows = load_table(MOBILE, 'cases.json', mobile)
    missing_by_frame = {r['key']:r for r in missed_rows}
    cohorts, _, fixtures = groups()
    captures = {f'{c.floor}/{c.id}':c for cohort in cohorts.values() for c in cohort}
    roster = {r['key']:r['sha256'] for r in rows}
    if (len(rows) != len(roster) or len(roster) != 56 or roster.keys() != captures.keys()
            or not all(len(table) == 56 and {r['key']:r['sha256'] for r in table} == roster
                       for table in (missed_rows, sam_rows))
            or not all(m['fixtures'] == fixtures and m['reference_sha256'] == source['reference_sha256']
                       for m in (source, mobile, misses))):
        raise ValueError('Unexpected fixture, frame roster, or camera inputs')
    checked(ARCHIVE, source['reference_sha256'])
    for folder in (SOURCE, MOBILE):
        checked(folder/'manifest.json', misses['source_manifests'][folder.name])
    targets, boundaries, cuts = [], [], []
    for row in rows:
        key = row['key']
        per_target, per_boundary, per_cut = inspect_frame(captures[key], row, missing_by_frame[key], source, mobile)
        targets.extend(per_target)
        boundaries.extend(per_boundary)
        cuts.extend(per_cut)
    summary = summarize(targets, boundaries, cuts)
    if summary['misses'] != misses['totals']['missed']:
        raise ValueError('Miss count changed')
    out.mkdir(parents=True, exist_ok=False)
    files = {'summary.json':summary, 'per-miss.json':targets, 'individual-cuts.json':cuts, 'contrast.json':boundaries}
    for name, contents in files.items():
        save(out/name, contents)
    save(out/'manifest.json', dict(run_id=out.name, code=revision(), version=source['version'],
         method='wall-recovery-single-v2', scope='Scoring diagnostic. Each extra cut is tested alone on an intact final parent.',
         source_manifests={p.name:digest(p/'manifest.json') for p in (SOURCE, MOBILE, MISSES)}, fixtures=fixtures,
         reference_sha256=digest(ARCHIVE), frame_count=len(rows), per_frame_input_sha256=roster,
         settings=dict(match_iou=PROTOCOL['iou'], stride=PROTOCOL['sample_stride'],
                       contrast='Median of the mean absolute RGB difference across adjacent 4x4-pixel sample cells.',
                       threshold_selection='Best balanced accuracy on development. Highest threshold breaks ties. Mixed boundaries excluded.',
                       sam='Highest whole-mask IoU per missed reference among all raw masks, selected masks, masks offered at the original depth vote and offers with intact final parents. Reference identifies opportunities only.'),
         artifacts={name:digest(out/name) for name in files},
         reproduce='python -m benchmark.afterpass_headroom --out runs/afterpass-headroom-replay'))
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs'/f'afterpass-headroom-{uuid4().hex[:12]}')
    run(parser.parse_args().out.resolve())
