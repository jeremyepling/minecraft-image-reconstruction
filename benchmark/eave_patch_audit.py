"""Test small enclosed mask fragments against surrounding surface and same-row background."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from benchmark import facade_inset_round as suite
from benchmark.texture_camera_eval import read, save
from benchmark.wall_texture_evaluate import metrics
from engine import eave_pinholes, inset_returns
from engine.normal_regions import _local_mask
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs

MODES = ('size_only', 'background', 'enclosed', 'guarded', 'underside_holes', 'fully_surrounded_eave')
REVIEW = Path('benchmark/reviews/shopfront-eave-01')
SETTINGS = dict(min_pixels=4, max_pixels=1024, ring_radius=3, min_wrap=.55,
                guarded_wrap=.80, background_fraction=.90, max_color_step=8, underside_y=.85, eave_aspect=4.)


def components(mask):
    """Group native half-open row spans by four-connectivity."""
    spans = row_runs(mask)
    parent = list(range(len(spans)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    above, current, previous_y = [], [], -2
    for i, (y, left, right) in enumerate(spans):
        if y != previous_y:
            above = current if y == previous_y + 1 else []
            current, previous_y = [], y
        for a, b, index in above:
            if a >= right:
                break
            if b > left:
                parent[root(i)] = root(index)
        current.append((left, right, i))
    groups = defaultdict(list)
    for i, span in enumerate(spans):
        groups[root(i)].append(span)
    return groups.values()


def fragments(mask, occupied):
    """Small pieces of support and fully enclosed, unassigned holes."""
    y, x = np.nonzero(mask)
    if not x.size:
        return
    left, top = int(x.min()), int(y.min())
    local = mask[top:y.max()+1, left:x.max()+1]
    height, width = local.shape
    for kind, layer in [('piece', local), ('hole', ~local)]:
        for spans in components(layer):
            area = sum(b-a for _, a, b in spans)
            if not SETTINGS['min_pixels'] <= area <= SETTINGS['max_pixels']:
                continue
            if kind == 'hole' and any(y in (0,height-1) or a==0 or b==width for y,a,b in spans):
                continue
            patch, x, y = _local_mask(np.asarray(spans))
            x, y = x+left, y+top
            if kind == 'hole' and np.any(occupied[y:y+patch.shape[0], x:x+patch.shape[1]][patch]):
                continue
            yield kind, patch, x, y


def proposals(image, parts):
    rgb = np.asarray(image)
    masks = [native_mask(p,image.size) for p in parts]
    if not masks:
        return [], masks
    occupied = np.logical_or.reduce(masks)
    sizes = [int(m.sum()) for m in masks]
    events = []
    for i, mask in enumerate(masks):
        ys,xs = np.nonzero(mask)
        eave = len(ys) and parts[i]['normal_camera'][1] >= SETTINGS['underside_y'] and (np.ptp(xs)+1) >= SETTINGS['eave_aspect']*(np.ptp(ys)+1)
        for kind, patch, left, top in fragments(mask,occupied):
            r = SETTINGS['ring_radius']
            pad = np.pad(patch,r)
            area = int(patch.sum())
            ring = np.asarray(Image.fromarray(pad.astype('uint8')).filter(ImageFilter.MaxFilter(2*r+1)),bool) & ~pad
            yy,xx = np.nonzero(ring)
            yy,xx = yy+top-r,xx+left-r
            valid = (yy>=0)&(xx>=0)&(yy<rgb.shape[0])&(xx<rgb.shape[1])
            yy,xx = yy[valid],xx[valid]
            y,x = np.nonzero(patch)
            y,x = y+top,x+left
            # Holes belong to the surrounding host. A detached piece requires a larger neighbor.
            hosts = [i] if kind=='hole' else [j for j in range(len(masks)) if i!=j and sizes[j]>=4*area]
            if not hosts or not len(yy):
                continue
            wrap, host = max((float(masks[j][yy,xx].mean()),j) for j in hosts)
            bg = eave_pinholes.row_background(rgb,y,x)
            if wrap < SETTINGS['min_wrap'] and (kind=='hole' or bg<SETTINGS['background_fraction']):
                continue
            near = masks[host][yy,xx]
            colors = rgb[y,x].astype(float)
            rim = rgb[yy[near],xx[near]].astype(float)
            step = None
            if len(rim):
                step = float(np.max(abs(np.quantile(colors,[.25,.50,.75],axis=0)
                                       - np.quantile(rim,[.25,.50,.75],axis=0))))
            guarded = wrap>=SETTINGS['guarded_wrap'] and step is not None and step<=SETTINGS['max_color_step']
            drop = kind=='piece' and bg>=SETTINGS['background_fraction']
            actions = dict(size_only='remove' if kind=='piece' else None,
                           background='remove' if drop else None,
                           enclosed='transfer' if wrap>=SETTINGS['min_wrap'] else None,
                           guarded='remove' if drop else 'transfer' if guarded and bg<.15 else None,
                           underside_holes='transfer' if kind=='hole' and eave and guarded and bg<.15 else None,
                           fully_surrounded_eave='transfer' if kind=='hole' and eave and wrap==1. and guarded and bg<.15 else None)
            events.append((dict(kind=kind, source=parts[i]['id'], host=parts[host]['id'],
                                box=[left,top,left+patch.shape[1],top+patch.shape[0]], pixels=area,
                                wrap=round(wrap,4), bg=round(bg,4), step=None if step is None else round(step,2),
                                actions=actions),i,host,y,x))
    return events,masks


def describe_change(parts, masks, events, mode, truth, size, before):
    changed = {}
    for row, src, dst, y, x in events:
        action = row['actions'][mode]
        if not action:
            continue
        if row['kind']=='piece':
            if src not in changed:
                changed[src] = masks[src].copy()
            changed[src][y,x] = False
        if action=='transfer':
            if dst not in changed:
                changed[dst] = masks[dst].copy()
            changed[dst][y,x] = True
    if not changed:
        return dict(gained=[], lost=[], unpaired_delta=0, paired_px=0, background_px=0, lost_coverage=0)
    result = [p if i not in changed else dict(p, support=dict(size=list(size),runs=row_runs(changed[i])))
              for i,p in enumerate(parts) if i not in changed or changed[i].any()]
    after = metrics(result,truth,size)
    old_union = np.logical_or.reduce(masks)
    new_union = np.logical_or.reduce([changed.get(i,m) for i,m in enumerate(masks)])
    return dict(gained=sorted(set(after['matched'])-set(before['matched'])),
                lost=sorted(set(before['matched'])-set(after['matched'])),
                unpaired_delta=after['unmatched_predictions']-before['unmatched_predictions'],
                paired_px=after['paired_pixels']-before['paired_pixels'],
                background_px=after['background_pixels']-before['background_pixels'],
                lost_coverage=int((old_union & ~new_union & (truth>0)).sum()))


def measure(arg):
    job, expected = arg
    split, frame, _, _ = job
    image, parts, _, camera = suite.ordinary_inputs(*job)
    if any(expected[k] for k in ('returns','texture','outlines')):
        parts, strict, phase, outlines, _ = inset_returns.detect(image,parts,'26.3',camera=camera)
        assert [len(v) for v in [strict,phase,outlines]]==[expected[k] for k in ['returns','texture','outlines']]
    events,masks = proposals(image,parts)
    chosen = [e[0] for e in events if e[0]['actions']['fully_surrounded_eave']]
    actual, actions = eave_pinholes.refine(image,parts)
    assert sorted((r['host'],r['box'],r['pixels']) for r in actions) == sorted((r['host'],r['box'],r['pixels']) for r in chosen)
    # Independently compare the production support bitmaps to the selected candidate.
    for i in {e[2] for e in events if e[0]['actions']['fully_surrounded_eave']}:
        expected_mask = masks[i].copy()
        for e,_,dst,y,x in events:
            if e['actions']['fully_surrounded_eave'] and i==dst:
                expected_mask[y,x] = True
        assert np.array_equal(expected_mask,native_mask(actual[i],image.size))
    # Sealed proposals; only now load recorded geometry.
    truth = suite.visible_map(split,frame)
    baseline = metrics(parts,truth,image.size)
    dominants = []
    for p in baseline['parts']:
        dominants.append(p['dominant'] if p['correct_if_dominant']>p['background'] else 0)
    for row, src, dst, y, x in events:
        ids = truth[y,x]
        row.update(background_px=int((ids==0).sum()), source_correct_px=int((ids==dominants[src]).sum()) if dominants[src] else 0,
                   host_correct_px=int((ids==dominants[dst]).sum()) if dominants[dst] else 0)
    scores = {k:describe_change(parts,masks,events,k,truth,image.size,baseline) for k in MODES}
    return dict(id=frame['id'],split=split,rgb_sha256=frame['files']['rgb.png'],
                label_map_sha256=sha256(truth.astype('<i4').tobytes()).hexdigest(),
                mask_sha256=sha256(json.dumps([(p['id'],p['support']) for p in parts],sort_keys=True).encode()).hexdigest(),
                baseline_matches=len(baseline['matched']),scores=scores,events=[e[0] for e in events])


def summary(rows):
    methods={}
    for mode in MODES:
        c=Counter()
        for row in rows:
            active=[e for e in row['events'] if e['actions'][mode]]
            c['actions']+=len(active)
            c['changed_views']+=bool(active)
            for key,value in row['scores'][mode].items():
                c[key]+=len(value) if isinstance(value,list) else value
        methods[mode]=dict(c)
    return dict(views=len(rows),methods=methods)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--shopfront-only',action='store_true')
    parser.add_argument('--all-cohorts',action='store_true')
    parser.add_argument('--out',type=Path,default=REVIEW/'cases.json')
    args=parser.parse_args()
    existing={r['id']:r for r in read(suite.ROOT/'benchmark/reviews/facade-inset-round-01/combined.json')['rows']}
    jobs=[(j,existing[j[1]['id']]) for j in suite.jobs([*suite.framed.SIZES,'shelter'] if args.all_cohorts else suite.framed.EVALUATION)
          if not args.shopfront_only or 'shopfront' in j[1]['id']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases=list(pool.map(measure,jobs))
    outcomes={k:summary(v) for k,v in [('fixed',[r for r in cases if r['split'] in suite.framed.EVALUATION]),('additional',[r for r in cases if r['split'] not in suite.framed.EVALUATION]),('shopfront',[r for r in cases if 'shopfront' in r['id']]),
                                        ('other_layouts',[r for r in cases if 'shopfront' not in r['id'] and r['split'] in suite.framed.EVALUATION])]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,dict(base='a95816d',settings=SETTINGS,cohorts=outcomes,cases=cases))
    print(outcomes)


if __name__=='__main__':
    main()
