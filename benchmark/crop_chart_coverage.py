"""Audit sampling windows for the split-only Surface Finder's matched surfaces."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, read, revision, save
from benchmark.mask_metrics import PROTOCOL, measure
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.surface_perimeter import _native_alpha
from engine.surface_sample import sample_surface
from runtime.mode import image_only


SOURCE = ROOT / 'benchmark/reviews/crop-split-01'
SOURCE_MANIFEST = 'b300caff403e517e09003376a112254bc1d72a87e3e52c1b53224c6a892f1850'


def chart_covered_pixels(alpha, bounds, quad):
    """Count native source-pixel centers inside both the visible mask and the sampling window."""
    yy, xx = np.nonzero(np.asarray(alpha))
    xx = xx + bounds[0] + .5
    yy = yy + bounds[1] + .5
    signs = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        signs.append((b[0]-a[0])*(yy-a[1]) - (b[1]-a[1])*(xx-a[0]))
    signed = np.asarray(signs)
    return int(np.count_nonzero(np.all(signed>=-1.e-8, axis=0) | np.all(signed<=1.e-8, axis=0)))


def sample_record(image, region, rid):
    alpha, bounds = _native_alpha(region['support'], image.size)
    sample = sample_surface(image, region)
    record = dict(reference_id=rid, proposal=region['id'], source_mask_samples=int(np.count_nonzero(alpha)),
                  chart_kind=None, chart_covered_samples=None, valid_output_pixels=0)
    if sample is not None:
        record.update(chart_kind=sample['chart_source'], chart_covered_samples=chart_covered_pixels(alpha, bounds, sample['quad']),
                      valid_output_pixels=int(sample['mask'].sum()))
    return record


@image_only()
def run(out):
    source = read(checked(SOURCE / 'manifest.json', SOURCE_MANIFEST))
    checked(ARCHIVE, source['reference_sha256'])
    cases = read(checked(SOURCE / 'cases.json', source['artifacts']['cases.json']))
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for group in cohorts.values() for c in group}
    roster = {c['key']: c['sha256'] for c in cases}
    if (suite['version'] != source['version'] or fixtures != source['fixtures'] or source['protocol'] != PROTOCOL
            or len(cases) != 56 or set(captures) != roster.keys()
            or {f['key']: f['sha256'] for f in source['frames']} != roster):
        raise ValueError('Fixture list or sampling protocol differs from the split-only run')
    frames = []
    for case in cases:
        capture = captures[case['key']]
        with Image.open(checked(capture.path('image'), case['sha256'])) as original:
            image = original.convert('RGB')
        path = case['predictions']
        parts = read_gzip(checked(SOURCE / path, source['artifacts'][path]))['split']
        _, reference = recorded(capture)
        _, pairs = measure(parts, reference, use_mask=True)
        if sorted(rid for _, rid in pairs) != case['scopes']['split']['matched']:
            raise ValueError(f"Matched surfaces changed for {case['key']}")
        rows = [sample_record(image, parts[i], rid) for i, rid in pairs]
        frames.append(dict(frame=case['key'], sha256=case['sha256'], matched=rows))
    records = [r for frame in frames for r in frame['matched']]
    usable = [r for r in records if r['valid_output_pixels']]
    coverage = np.array([r['chart_covered_samples'] / r['source_mask_samples'] for r in usable])
    summary = dict(matched=len(records), usable=len(usable), no_chart=sum(r['chart_kind'] is None for r in records),
                   empty_sample=sum(r['chart_kind'] is not None and not r['valid_output_pixels'] for r in records),
                   usable_chart_kinds=dict(Counter(r['chart_kind'] for r in usable)),
                   usable_below_90_percent=int((coverage < .9).sum()),
                   usable_below_80_percent=int((coverage < .8).sum()))
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, dict(run_id='crop-coverage-01', code=revision(), source_manifest=SOURCE_MANIFEST,
                   source_run=SOURCE.relative_to(ROOT).as_posix(), method='plane-recovery-crop-splits',
                   game_version=source['version'], reference_sha256=source['reference_sha256'],
                   sample_stride_px=1, scope='Eligible, matched surface appearances only.',
                   measure='Fraction of native source-mask pixels inside the chart, before filtering the interpolation taps. '
                           'A usable sample contains at least one accepted output pixel.',
                   summary=summary, frames=frames))
    print(summary)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True, help='Fresh path for the JSON report')
    run(cli.parse_args().out)
