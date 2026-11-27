"""Fit on Pillars/Viaduct and run a frozen source-only crop chooser on six Sky frames."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from benchmark.sky_crop_options import ACTIVE, OPTIONS, OUTPUT, SKY, WORKTREES, fields, native
from engine.floating_sky import sky_support


SAM = ('mobile-box','mobile-points','sam-b-box','sam-b-points','sam-b-zoom')
MODEL_RUN = WORKTREES/'sky-crop-models/runs/sky-crop-models'
DEV = ('train-pillars-01-marble-low-left','train-viaduct-02-riverstone-high-right')
FEATURES = ('log_area','fill','sky','normal_spread','inverse_depth_residual','boundary_contrast',
            'mask_iou','lost_share','new_share','mean_consensus','family','changed')


def paths(folder):
    return {**{name:folder/f'{name}.json' for name in OPTIONS}, **{n:MODEL_RUN/f'{n}.json' for n in SAM}}


def overlap(a, b):
    hit = np.count_nonzero(a & b)
    return hit / max(1, np.count_nonzero(a) + np.count_nonzero(b) - hit)


def describe(mask, normal, depth, rgb, sky):
    y, x = np.nonzero(mask)
    if not len(x):
        return np.zeros(6)
    idx = np.linspace(0, len(x)-1, min(512, len(x)), dtype=int)
    yy, xx = y[idx], x[idx]
    normals = normal[yy,xx]
    design = np.column_stack((yy, xx, np.ones_like(yy)))
    z = depth[yy,xx]
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    crop = mask[y.min():y.max()+1, x.min():x.max()+1]
    # Compare immediate inside and outside pixels across horizontal and vertical mask edges.
    boundary = []
    for axis in (0, 1):
        yy, xx = np.nonzero(np.diff(np.pad(crop, 1), axis=axis))
        yy += y.min() - 1
        xx += x.min() - 1
        dy, dx = int(axis == 0), int(axis == 1)
        ok = (xx >= 0) & (yy >= 0) & (xx+dx < mask.shape[1]) & (yy+dy < mask.shape[0])
        xx, yy = xx[ok], yy[ok]
        if len(xx) > 1024:
            take = np.linspace(0, len(xx)-1, 1024, dtype=int)
            xx, yy = xx[take], yy[take]
        boundary.extend(np.max(np.abs(rgb[yy,xx]-rgb[yy+dy,xx+dx]), axis=1))
    return np.array((np.log1p(len(x)), crop.mean(), sky[y,x].mean(),
                     1 - np.linalg.norm(normals.mean(axis=0)),
                     np.mean(abs(z - design @ coef)), np.mean(boundary)/255. if boundary else 0.))


def make_features(output):
    packets = {n: read(path)['frames'] for n, path in paths(output).items()}
    result = {}
    for folder in sorted((SKY/'baseline/sky').glob('*/*')):
        case = folder.name
        path = SKY/'images'/case/'sky.png'
        if digest(path) != read(folder/'receipt.json')['source_sha256']:
            raise ValueError(f'Wrong source {case}')
        image = Image.open(path).convert('RGB')
        rgb = np.asarray(image, np.int16)
        normals, depth, _ = fields(folder, image.size)
        sky = sky_support(image)
        features = result[case] = {}
        for part in read(ACTIVE/case/'source-handoff/surfaces.json')['planes']:
            pid = part['id']
            base = native(part['support'], image.size)
            base_desc = describe(base, normals, depth, rgb, sky)
            # All options for this proposal share one crop ROI for their mutual intersections.
            masks = {'baseline':base}
            for name, packet in packets.items():
                s = packet[case].get(pid)
                masks[name] = base if s is None else native(s, image.size)
            y,x = np.nonzero(np.logical_or.reduce(list(masks.values())))
            bbox = np.s_[y.min():y.max()+1,x.min():x.max()+1]
            local = {n:m[bbox] for n,m in masks.items()}
            # Balance the five closely related learned prompts and the independent option families.
            def consensus(name):
                models = np.mean([overlap(local[name],local[n]) for n in SAM])
                geometric = np.mean([overlap(local[name],local[n]) for n in OPTIONS])
                return (models + geometric) / 2.
            base_vote = consensus('baseline')
            per_mask = features[pid] = {}
            for name, m in masks.items():
                if name == 'baseline':
                    continue
                changed = not np.array_equal(base, m)
                delta = describe(m, normals, depth, rgb, sky) - base_desc if changed else np.zeros(6)
                delta = np.clip(delta, -4., 4.)
                b, c = local['baseline'], local[name]
                common = (b & c).sum()
                rel = [1-overlap(b,c), 1-common/max(1,b.sum()), 1-common/max(1,c.sum()),
                       consensus(name) - base_vote]
                family = [int(name == n) for n in packets]
                per_mask[name] = dict(changed=changed, values=np.r_[delta,rel,family,1.].tolist())
        print(f'Features {case}',flush=True)
    save(output/'features.json', dict(frames=result, feature_groups=FEATURES,
                                     option_sha256={k:digest(v) for k,v in paths(output).items()}))


def observations(features, cases):
    """Load supervised records for precisely the requested training buildings."""
    x, y = [], []
    for case in cases:
        arms = case['arms']
        for base in arms['baseline']['proposals']:
            pid = base['id']
            for name, feat in features[case['id']][pid].items():
                if feat['changed']:
                    cand = next(item for item in arms[name]['proposals'] if item['id'] == pid)
                    x.append(feat['values'])
                    y.append(cand['iou'] - base['iou'])
    return np.asarray(x), np.asarray(y)


def fit(features, cases, penalty):
    x, y = observations(features,cases)
    scale = np.maximum(np.sqrt(np.mean(x**2, axis=0)),.01)
    x /= scale
    beta = np.linalg.solve(x.T @ x + penalty*np.identity(x.shape[1]), x.T @ y)
    return (beta / scale).tolist()


def choose(features, beta, margin):
    result = {}
    for pid, options in features.items():
        accepted = [(np.dot(feat['values'],beta),n) for n,feat in options.items() if feat['changed']]
        score, best = max(accepted, default=(0.,'baseline'))
        result[pid] = best if score > margin else 'baseline'
    return result


def train(output, score_path):
    features = read(output/'features.json')
    cases = {c['id']:c for c in read(score_path)['cases'] if c['id'] in DEV}
    if set(cases) != set(DEV):
        raise ValueError('Both development buildings are required')
    trials = []
    for penalty in (4.,32.,128.):
        for threshold in (.0,.02,.05):
            deltas = []
            for train_case, valid in (DEV, DEV[::-1]):
                beta = fit(features['frames'], [cases[train_case]], penalty)
                choices = choose(features['frames'][valid],beta,threshold)
                arms = cases[valid]['arms']
                delta = [row['iou'] - base['iou'] for base in arms['baseline']['proposals']
                         for row in arms[choices[base['id']]]['proposals'] if row['id'] == base['id']]
                deltas.append(float(np.mean(delta)))
            trials.append(dict(penalty=penalty,threshold=threshold,building_deltas=deltas))
    # Favor the rule with the best transfer to the weaker of the two training buildings.
    best = max(trials, key=lambda t:(min(t['building_deltas']),sum(t['building_deltas']),t['penalty']))
    model = dict(train_ids=DEV, loss='per-crop fixed-baseline-target IoU gain', trials=trials, **best,
                 coefficients=fit(features['frames'],list(cases.values()),best['penalty']),
                 options=list(paths(output)),option_sha256=features['option_sha256'])
    save(output/'frozen.json',model)


def emit(output, choices, name):
    support = {n:read(v)['frames'] for n,v in paths(output).items()}
    frames = {case:{p:support[n][case][p] for p,n in by_plane.items()
                    if n != 'baseline' and p in support[n][case]} for case,by_plane in choices.items()}
    save(output/f'{name}.json',dict(frames=frames))
    save(output/f'{name}-choices.json',dict(frames=choices))
    print(name, dict(Counter(n for row in choices.values() for n in row.values())))


def infer(output):
    model = read(output/'frozen.json')
    features = read(output/'features.json')
    current = {k:digest(v) for k,v in paths(output).items()}
    if model['option_sha256'] != current or features['option_sha256'] != current:
        raise ValueError('Frozen model option provenance mismatch')
    emit(output, {case:choose(group,model['coefficients'],model['threshold'])
                  for case,group in features['frames'].items()},'learned')


def oracle(output, score_path):
    choices = {}
    for case in read(score_path)['cases']:
        arms = case['arms']
        by_id = {n:{r['id']:r['iou'] for r in data['proposals']} for n,data in arms.items()}
        choices[case['id']] = {pid:max(by_id,key=lambda n:by_id[n][pid]) for pid in by_id['baseline']}
    emit(output,choices,'oracle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('features','train','infer','oracle'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--scores', type=Path)
    args = parser.parse_args()
    if args.action == 'features':
        make_features(args.output)
    elif args.action == 'train':
        train(args.output,args.scores)
    elif args.action == 'infer':
        infer(args.output)
    else:
        oracle(args.output,args.scores)


if __name__ == '__main__':
    main()
