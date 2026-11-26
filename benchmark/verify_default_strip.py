"""Verify the promoted strip cleanup against the pre-promotion masks and scored cases."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark.attached_strip_round import evaluate, summarize
from benchmark.camera_round import ROOT, read, revision, save
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE
from benchmark.wall_join_test import checked
from engine.strip_neighbor import clean_surfaces
from runtime.mode import image_only, mark_image_only


APPROVED = ROOT / 'benchmark/reviews/strip-neighbor-01'
SIGNATURES = ROOT / 'benchmark/reviews/default-strip-neighbor-01/approved-regions.json'


def region_signature(parts):
    data = json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def approved_row(key):
    manifest = read(APPROVED / 'manifest.json')
    rows = read(checked(APPROVED / 'cases.json', manifest['artifacts']['cases.json']))
    return next(r for r in rows if r['key'] == key)


def verify_frame(job):
    capture, source, sha, expected, signature = job

    trim_record = {}

    def run_filter(photo, parts):
        actual, stats = clean_surfaces(photo, parts)
        if region_signature(actual) != signature:
            raise ValueError('Saved pre-promotion mask or proposal property changed on '+source['key'])
        edits = expected['removed']['neighbor']
        if stats['changed_surfaces'] != len(edits) or stats['removed_pixels'] != sum(d['removed'] for d in edits):
            raise ValueError('Default strip removal statistics differ on '+source['key'])
        trim_record.update(pixels=stats['removed_pixels'], surfaces=stats['changed_surfaces'])
        return actual, []

    result = evaluate((capture, source, sha), refiners=(('neighbor', run_filter),))
    for group in ('scores', 'scopes', 'sizes', 'size_changes'):
        if result[group]['neighbor'] != expected[group]['neighbor']:
            raise ValueError(f"Promoted {group} differs on {source['key']}")
    result['verified_trim'] = trim_record
    return result


@image_only()
def verify_suite(workers):
    source = read(checked(BASE/'manifest.json', BASE_SHA))
    gold = read(APPROVED/'manifest.json')
    checked(ARCHIVE, gold['reference_sha256'])
    for field in ('frames', 'fixtures', 'protocol', 'reference_sha256', 'splits', 'version'):
        if source[field] != gold[field]:
            raise ValueError('Source and approved suite disagree on '+field)
    fixtures, suite, hashes = groups()
    if hashes != gold['fixtures'] or suite['version'] != gold['version']:
        raise ValueError('Current capture library differs')
    captures = {c.floor+'/'+c.id:c for group in fixtures.values() for c in group}
    rows = read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    comparison = read(checked(APPROVED/'cases.json', gold['artifacts']['cases.json']))
    expected = {r['key']:r for r in comparison}
    signatures = read(SIGNATURES)['frames']
    if len(rows)!=56 or captures.keys()!=expected.keys() or signatures.keys()!=expected.keys():
        raise ValueError('Missing verification inputs')
    jobs = [(captures[r['key']],r,source['artifacts'][r['predictions']],expected[r['key']],signatures[r['key']])
            for r in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        results = list(pool.map(verify_frame,jobs))
    scores = summarize(results,('current','neighbor'))['all']['neighbor']
    return dict(code=revision(),source_run='strip-neighbor-01',approved_revision=read(SIGNATURES)['revision'],
                scope='Archived inputs to the production strip filter; full upstream inference is tested separately.',
                frames=len(results),all_ordered_proposal_records_equal=True,all_per_frame_scores_and_sizes_equal=True,
                changed_views=sum(r['verified_trim']['surfaces']>0 for r in results),
                changed_surfaces=sum(r['verified_trim']['surfaces'] for r in results),
                removed_native_pixels=sum(r['verified_trim']['pixels'] for r in results),
                eligible=scores['eligible'],matched=scores['matched'],sizes=scores['sizes'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=6)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    save(args.out,verify_suite(args.workers))
