"""Rank fixed whole-structure layouts and map each representative through the measured pipeline."""
import argparse
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import re

from benchmark import framed_depth_eval as framed
from benchmark.texture_camera_eval import checked, read, save


def view_f1(metrics):
    return Fraction(2 * metrics['correct'], metrics['target'] + metrics['placed'])


def layout_name(frame):
    name = frame.get('layout') or frame.get('source_id') or frame['id']
    name = re.sub(r'^(train|test|holdout|heldout|confirm)-', '', name)
    name = re.sub(r'-(oblique-|high-|low-)?(left|right)$', '', name)
    name = re.sub(r'-oblique$', '', name)
    name = re.sub(r'([a-z])([0-9])', r'\1 \2', name)
    name = name.replace('-', ' ').title()
    if 'benchmark-open-' in frame['id']:
        name += ' (open)'
    return name


def inputs(run):
    summary = read(run / 'summary.json')
    rows = read(checked(run / 'cases.json', summary['cases_sha256']))
    scored = {(r['split'], r['id']): r for r in rows if r['split'] in framed.EVALUATION}
    assert len(scored) == 104
    frames = {}
    shapes = defaultdict(list)
    for split, frame, prior in framed.cases(framed.EVALUATION):
        key = split, frame['id']
        row = scored[key]
        assert frame['files']['rgb.png'] == row['rgb_sha256']
        _, occupied, _, _ = framed.references(split, frame, framed.NATIVE_INPUTS.get(split),
            framed.TRAINING_DATA, framed.DEPTH, dict(native=framed.native.BUILDS,
            old=framed.transfer.BUILDS, fresh=framed.transfer.BUILDS))
        shape = framed.transfer.shape_key(occupied)
        m = row['relative_geometry']
        view = dict(split=split, id=frame['id'], name=layout_name(frame), rgb_sha256=row['rgb_sha256'],
                    pack_sha256=row['pack_sha256'], f1=float(view_f1(m)),
                    **{k:m[k] for k in ('correct','target','placed','extras','missed')})
        frames[key] = (frame, prior, row)
        shapes[shape].append(view)
    layouts = []
    for shape, views in shapes.items():
        rep = min(views, key=lambda v: (view_f1(v), v['id']))
        pooled = {k:sum(v[k] for v in views) for k in ('correct','target','placed','extras','missed')}
        layouts.append(dict(name=rep['name'], shape_sha256=shape, frames=len(views), **pooled,
                            f1=float(view_f1(pooled)), representative=rep,
                            frame_ids=sorted(v['id'] for v in views)))
    layouts.sort(key=lambda r:(view_f1(r),r['shape_sha256']))
    for i, r in enumerate(layouts, 1):
        r['rank'] = i
        if i <= 10:
            r['sheet'] = f'structure-{i:02}.jpg'
    views = sorted((v for vs in shapes.values() for v in vs), key=lambda v:(view_f1(v),v['id']))
    return dict(case_count=len(scored), distinct_layouts=len(shapes), cases_sha256=summary['cases_sha256'],
                control=summary['control'], candidate=summary.get('candidate'), revision=summary['revision'],
                layouts=layouts[:10], all_layouts=layouts, lowest_views=views[:10]), frames


def main(args):
    ranking, frames = inputs(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'ranking.json', ranking)
    if not args.ranks_only:
        from benchmark.roof_span_worst_sheet import render
        render(args.run, args.out, ranking, frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ranks-only', action='store_true')
    main(parser.parse_args())
