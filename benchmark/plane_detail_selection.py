"""Reproduce the staged Planarity experiment and archive compact per-picture scores."""
import argparse
from pathlib import Path

from benchmark import detail_round, plane_detail, plane_detail_diagnostic
from benchmark.camera_round import ROOT, read, save, digest

ROUNDS = (
    ('dev-plane-v1','development',['stride4','stride2']),
    ('dev-plane-v2','development',['add2','add2-t2500']),
    ('dev-plane-control','development',['add4']),
    ('plane-grid-control-56','all',['add4']),
)


def pack(work, out):
    trials = {}
    for name,split,options in ROUNDS:
        path = work/name
        manifest = read(path/'manifest.json')
        scores = read(path/'scores.json')
        cases = read(path/'cases.json')
        if manifest['selection'] != split or manifest['methods'] != ['current',*['plane_detail:'+m for m in options]]:
            raise ValueError('Unexpected experiment selection')
        for source in ('scores.json','cases.json'):
            if digest(path/source) != manifest['artifacts'][source]:
                raise ValueError('Archived experiment changed')
        rows = []
        for r in cases:
            methods = {}
            for m in manifest['methods']:
                native = r['native'][m]
                methods[m] = dict(matched=r['scores'][m]['matched'],sizes=r['sizes'][m],
                                  **native)
            rows.append(dict(key=r['key'],sha256=r['sha256'],results=methods))
        trials[name] = dict(code=manifest['code'],method_sha256=manifest['candidate_sha256'],
            settings={s:plane_detail.provenance(s) for s in options},frames=manifest['frames'],
            protocol=manifest['protocol'],fixtures=manifest['fixtures'],capture=manifest['capture'],
            score_sha256=manifest['artifacts']['scores.json'],worker_count=manifest['workers'],
            score_semantics=manifest['native_metric'],totals=scores,cases=rows)
    save(out/'trials.json',dict(trials=trials,stage_diagnostic_sha256=digest(out/'diagnostic.json'),
        fields={p:digest(ROOT/p) for p in {p for _,_,settings in ROUNDS for s in settings
            for p in plane_detail.provenance(s)['field_manifests']}},
        selection='Two-way snapping dropped many true pixels in development. Addition-only gains were positive. '
                  'Freeze add2 with 1600 tokens over add2 with 2500 before the full comparison. Then test add4 '
                  'as an explicitly labeled ablation, using the identical add-only gate; no further settings selected from regression.',
        reproduce='python -m benchmark.plane_detail_selection --out runs/plane-detail-replay'))


def replay(out,workers):
    out.mkdir(parents=True,exist_ok=False)
    for name,split,settings in ROUNDS:
        detail_round.run(out/name,workers,split,['plane_detail:'+s for s in settings],False)
    plane_detail_diagnostic.run('all',[1600,2500],out/'diagnostic.json')
    pack(out,out)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',required=True,type=Path)
    cli.add_argument('--workers',type=int,default=4)
    args=cli.parse_args()
    replay(args.out.resolve(),args.workers)
