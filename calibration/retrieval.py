"""Rendered exemplar similarity and block ranking shared by calibration and reconstruction."""
from collections import Counter

import numpy as np
from PIL import Image

METHODS = ('color', 'aligned', 'mottling', 'combined', 'multiscale')


def features(tiles):
    # Trim the outer pixel where neighboring cells bleed through interpolation.
    rgb = tiles[:, 1:-1, 1:-1].astype(np.float32) / 255
    mean = rgb.mean(axis=(1, 2))
    light = rgb @ np.array([.213, .715, .072], np.float32)
    patches = (rgb - mean[:, None, None]).reshape(len(rgb), -1)
    patches /= np.maximum(.025, np.sqrt((patches**2).mean(1, keepdims=True)))
    increments = []
    for lag in (1, 2, 4):
        for dy, dx in ((lag, 0), (0, lag), (lag, lag), (lag, -lag)):
            y, x = light.shape[1:]
            a = light[:, :y-dy, max(0, -dx):min(x, x-dx)]
            b = light[:, dy:, max(0, dx):min(x, x+dx)]
            diff = np.abs(a-b)
            increments.extend([diff.mean((1, 2)), np.sqrt((diff**2).mean((1, 2)))])
    coarse=light.reshape(len(light),7,2,7,2).mean(axis=(2,4)).reshape(len(light),49)
    coarse-=coarse.mean(axis=1,keepdims=True)
    coarse/=np.maximum(.025,coarse.std(axis=1,keepdims=True))
    return mean, patches, np.log1p(40*np.stack(increments, axis=1)), coarse


def mse(a, b):
    return np.maximum(0, (a*a).mean(1)[:, None] + (b*b).mean(1) - 2*(a @ b.T)/a.shape[1])


def distances(query, reference):
    color, spatial, shape, coarse = (mse(a, b) for a,b in zip(query,reference))
    # Use common-exposure color plus complementary descriptors. All weights
    # are fixed before validation. Each column denotes one training exemplar.
    aligned = 12*color + .16*spatial
    mottling = 12*color + shape
    return dict(color=color, aligned=aligned, mottling=mottling, combined=aligned+.7*shape,
                multiscale=12*color+.08*spatial+.08*coarse+.7*shape)


def rank(query, train, labels, method):
    """Rank distinct block IDs. Multiple exemplars of one ID get one vote."""
    order = np.argsort(labels, kind='stable')
    names, starts = np.unique(np.asarray(labels)[order], return_index=True)
    if len(names) < 2:
        raise ValueError('Need at least two labeled classes')
    for start in range(0, len(query), 96):
        d = distances(features(query[start:start+96]), train)[method]
        per_class = np.minimum.reduceat(d[:, order], starts, axis=1)
        best = np.argsort(per_class, axis=1, kind='stable')[:, :5]
        for idx, row in zip(best, per_class):
            gap = (row[idx[1]]-row[idx[0]]) / max(row[idx[1]], 1e-6)
            yield names[idx].tolist(), row[idx].tolist(), float(gap)


def measure(train, query, methods, out_predictions=False):
    refs, labels = train
    queries, truth = query
    samples = []
    for batch in range(0,len(queries),96):
        ds = distances(features(queries[batch:batch+96]), refs)
        for method in methods:
            # Nearest exemplar wins. Stable sorting makes ties reproducible.
            # Resolve an exact tie by class name, as in rank().
            order = np.argsort(labels,kind='stable')
            samples.extend((method,str(actual),str(labels[order[i]])) for actual,i in
                           zip(truth[batch:batch+96], ds[method][:,order].argmin(1)))
    report = {}
    for method in methods:
        rows = [(t,p) for m,t,p in samples if m==method]
        totals = Counter(t for t,p in rows)
        correct = Counter(t for t,p in rows if t==p)
        report[method] = dict(n=len(rows), correct=sum(correct.values()),
            macro=float(np.mean([correct[k]/n for k,n in totals.items()])),
            by_block={k:[correct[k],n] for k,n in sorted(totals.items())})
        if out_predictions:
            report[method]['predictions']=rows
    return report


def degrade(tiles,size,gains):
    images=[]
    for tile in tiles:
        small=Image.fromarray(tile).resize((size,size),Image.Resampling.BOX)
        soft=np.asarray(small.resize((16,16),Image.Resampling.BILINEAR))
        images.append(np.uint8(np.clip(soft*np.array(gains),0,255)))
    return np.stack(images)
