"""Score independent neighboring-mask support for a photo-supported tail trim."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import attached_strip, strip_evidence, strip_neighbor
from benchmark.attached_strip_round import evaluate, refine_evidence, summarize
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.mask_metrics import PROTOCOL
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE
from benchmark.wall_join_test import checked
from runtime.mode import image_only, mark_image_only

METHODS = ('current', 'evidence', 'neighbor')
PREVIOUS = ROOT / 'benchmark/reviews/attached-strip-01'
PREVIOUS_SHA = '75054c1bfbe65f9601885b235b63e04fc0494f2704045c5b773fcd5d96ddc85e'


def score(job):
    capture, row, sha, previous = job
    result = evaluate((capture, row, sha),
                      refiners=(('evidence', refine_evidence), ('neighbor', strip_neighbor.refine)))
    for group in ('scores', 'scopes', 'sizes', 'size_changes', 'removed'):
        for method in ('current', 'evidence'):
            if result[group].get(method) != previous[group].get(method):
                raise ValueError(f"Frozen {group} changed for {method} on {row['key']}")
    return result


@image_only()
def run(out, workers):
    source = read(checked(BASE / 'manifest.json', BASE_SHA))
    prior = read(checked(PREVIOUS / 'manifest.json', PREVIOUS_SHA))
    rows = read(checked(BASE / 'cases.json', source['artifacts']['cases.json']))
    replay = read(checked(PREVIOUS / 'cases.json', prior['artifacts']['cases.json']))
    checked(ARCHIVE, source['reference_sha256'])
    cohorts, suite, fixtures = groups()
    captures = {c.floor+'/'+c.id:c for cohort in cohorts.values() for c in cohort}
    previous_rows = {r['key']:r for r in replay}
    for field in ('protocol','fixtures','version','frames','reference_sha256','splits'):
        if source[field] != prior[field]:
            raise ValueError('Previous comparison used different '+field)
    if source['fixtures']!=fixtures or source['protocol']!=PROTOCOL or suite['version']!=source['version']:
        raise ValueError('Inputs do not match frozen suite')
    frames = {r['key']:r['sha256'] for r in source['frames']}
    if (frames!={r['key']:r['sha256'] for r in rows} or frames!={r['key']:r['sha256'] for r in replay}
            or len(frames)!=56 or captures.keys()!=frames.keys()):
        raise ValueError('Unexpected fixture inventory')
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(captures[r['key']],r,source['artifacts'][r['predictions']],previous_rows[r['key']]) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results = list(pool.map(score,jobs))
    scores = summarize(results,METHODS)
    save(out/'cases.json',results)
    save(out/'scores.json',scores)
    from benchmark.strip_neighbor_sheet import render
    render(out)
    modules = ('attached_strip','attached_strip_round','strip_evidence','strip_neighbor','strip_neighbor_round','strip_neighbor_sheet')
    manifest = dict(run_id=out.name,code=revision(),version=suite['version'],fixtures=fixtures,
        frames=source['frames'],splits=source['splits'],capture=source['capture'],protocol=PROTOCOL,
        reference_sha256=source['reference_sha256'],source_manifest='benchmark/reviews/sky-gap-01/manifest.json',
        source_manifest_sha256=BASE_SHA,source_code=source['code'],source_model_timing=source['source_model_timing'],
        previous_comparison='benchmark/reviews/attached-strip-01/manifest.json',previous_manifest_sha256=PREVIOUS_SHA,
        replay_check='Compared current and photo-only per-frame scores, scopes, sizes, exact reference changes and native removals.',
        predictions='gap',new_inference_s=0,workers=workers,methods=METHODS,
        settings=dict(shape=attached_strip.SETTINGS,evidence=strip_evidence.SETTINGS,neighbor=strip_neighbor.SETTINGS),
        module_sha256={m:digest(ROOT/('engine' if m in ('attached_strip','strip_evidence','strip_neighbor') else 'benchmark')/(m+'.py')) for m in modules},
        rule='Freeze before regression. Keep a new default only if it fixes the beam, protects the real wall and thin face, preserves prior matches and counts, and avoids collateral damage on the suite.',
        native_metric=prior['native_metric'],
        reproduce='python -m benchmark.strip_neighbor_round --out runs/strip-neighbor-replay',
        artifacts={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'manifest.json',manifest)
    for method, rec in scores['all'].items():
        print(method,rec['matched'],rec['sizes'],rec.get('removed',{}))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=6)
    args=cli.parse_args()
    run(args.out.resolve(),args.workers)
