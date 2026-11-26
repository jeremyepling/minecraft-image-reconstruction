"""Replay the frozen material experiment with main's verified photo-edge surface masks."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

from PIL import Image

SOURCE_REVISION = 'a12741dd1c26a632c26181e31116641cf345836d'
SOURCE_MANIFEST = 'd8a2c401d28c546762bcef38ed731c4074120f3750f2998a5b7e797fcb6e9a43'
SUITE_VERIFICATION = '46ad81e67a3420d555637c278a519199a344978f1ea008a3e3ba33d62acf75c8'
FINAL_MASKS = 'edge_detail:native4-stable-counts'


def use_main_engine(checkout):
    """Import production engine modules from the checked checkout in every replay process."""
    import engine
    from runtime.mode import mark_image_only
    if any(name.startswith('engine.') for name in sys.modules):
        raise RuntimeError('Run the current-engine replay in a separate Python process')
    engine.__path__.insert(0,str(checkout/'engine'))
    mark_image_only()


def check_downstream(checkout):
    """Pin the core downstream differences with the experiment revision."""
    from benchmark.camera_round import ROOT
    from benchmark.wall_join_test import checked
    # Current main added an off-grid contrast guard in native texel detection.
    checked(checkout/'engine/surface_pitch.py','8d25d962352b9b2756e2e7205cf86c55df38985b4aaa7e13f670816da04cd802')
    modules = ('geometry','surface_sample','masked_count','surface_dimensions','surface_tiles',
               'block_face_groups','blocker_shape','face_evidence','atlas_count','surface_model')
    for name in modules:
        def syntax(root):
            code = ast.parse((root/'engine'/(name+'.py')).read_text(encoding='utf8'))
            if name == 'surface_model':
                # The newer entry point applies photo edges before _count_regions.
                nodes = [n for n in code.body if getattr(n,'name','') in ('_count_regions','canonical','lens')]
                return [ast.dump(n) for n in nodes]
            return ast.dump(code)
        if syntax(ROOT) != syntax(checkout):
            raise ValueError('Downstream implementation differs from pinned experiment: '+name)


def score_current(job):
    from benchmark.camera_round import read
    from benchmark.phase_material_round import evaluate
    from benchmark.surface_reference import recorded
    from benchmark.wall_join_test import checked,read_gzip
    cap,row,archive,sha,check = job
    with Image.open(checked(cap.path('image'),row['sha256'])) as src:
        image = src.convert('RGB')
    parts = read_gzip(checked(archive,sha))[FINAL_MASKS]
    def reference():
        return recorded(cap)[1], read(cap.path('labels'))['faces']
    cohort = {'development':'development','previously_reviewed_regression':'regression'}[row['cohort']]
    result = evaluate(row['key'],row['sha256'],row['layout'],cohort,image,parts,reference)
    if result['matched'] != check['matched']:
        raise ValueError('Verified production match count changed on '+row['key'])
    return result


def run(checkout,out,workers):
    use_main_engine(checkout)
    from benchmark.camera_round import digest,read,save
    from benchmark.normal_round import groups
    from benchmark.phase_material_round import METHODS,RULE,matcher,totals
    from benchmark.surface_reference import ARCHIVE
    from benchmark.wall_join_test import checked
    source = checkout/'benchmark/reviews/edge-detail-stable-01'
    gold = read(checked(source/'manifest.json',SOURCE_MANIFEST))
    proof = read(checked(checkout/'benchmark/reviews/default-photo-edges-01/suite.json',SUITE_VERIFICATION))
    if proof['source_manifest_sha256']!=SOURCE_MANIFEST or not proof['all_ordered_proposal_records_equal']:
        raise ValueError('Expected the verified production mask archives')
    check_downstream(checkout)
    checked(ARCHIVE,gold['reference_sha256'])
    rows = read(checked(source/'cases.json',gold['artifacts']['cases.json']))
    cohorts,_,fixtures = groups()
    captures = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    checks = {r['frame']:r for r in proof['cases']}
    if (len(rows)!=56 or fixtures!=gold['fixtures'] or
            {r['key']:r['sha256'] for r in rows}!={r['key']:r['sha256'] for r in gold['frames']} or
            set(captures)!=set(checks) or set(checks)!={r['key'] for r in rows}):
        raise ValueError('Different evaluation fixture')
    jobs = [(captures[r['key']],r,source/r['predictions'],gold['artifacts'][r['predictions']],checks[r['key']])
            for r in rows]
    out.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers,initializer=use_main_engine,initargs=(checkout,)) as pool:
        results = list(pool.map(score_current,jobs))
    save(out/'cases.json',results)
    summaries = {scope:totals([f for f in results if scope in ('all',f['cohort'],f['layout'])])
                 for scope in ['all','development','regression',*sorted({f['layout'] for f in results})]}
    save(out/'scores.json',summaries)
    provenance = {m:matcher(m).provenance.copy() for m in METHODS}
    for row in provenance.values():
        row['source_sha256'] = {m:digest(checkout/'engine'/(m+'.py')) for m in row['source_sha256']}
    save(out/'manifest.json',dict(production_revision=SOURCE_REVISION,verification_sha256=SUITE_VERIFICATION,
         source_manifest_sha256=SOURCE_MANIFEST,reference_sha256=digest(ARCHIVE),fixtures=fixtures,
         source_method=FINAL_MASKS,rule=RULE,matchers=provenance,
         native_pitch_sha256=digest(checkout/'engine/surface_pitch.py'),
         runner_sha256=digest(Path(__file__)),evaluator_sha256=digest(Path(__file__).with_name('phase_material_round.py')),
         frames=[{k:f[k] for k in ('key','sha256','layout','cohort','outline_signature')} for f in results],
         artifacts={n:digest(out/n) for n in ('scores.json','cases.json')}))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkout',type=Path,required=True,help='Checkout containing the verified main photo-edge suite')
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=4)
    args=cli.parse_args()
    run(args.checkout.resolve(),args.out.resolve(),args.workers)
