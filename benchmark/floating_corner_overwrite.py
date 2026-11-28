"""Audit accepted shared-corner replacements against the saved pre-corner atlas charts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.floating_bridges_report import summarize
from benchmark.floating_extra_budget import LOCATION, combined, paths
from benchmark.shared_corner_rows import DENSE
from benchmark.full_block_lattice import atlas
from engine.surface_perimeter import _native_alpha

OUT = ROOT / 'runs/corner-overwrite'
REVIEW = ROOT / 'benchmark/reviews/floating-corner-overwrite-01'
BASE = WORKTREES / 'floating-extra-isolation/runs/extra-isolation'
FREEZE = WORKTREES / 'floating-depth-refit/benchmark/reviews/floating-depth-refit-01/conditions.json.gz'
ARCHIVE = WORKTREES / 'floating-extra-isolation/benchmark/reviews/floating-extra-isolation-01'
T1 = 'test-twinspire2-earth-high-right'
T1ROOT = WORKTREES / 'floating-occlusion-fragments/runs'
@lru_cache(None)
def sources(panel):
    return paths(panel)[0]


@lru_cache(None)
def receipts(panel):
    return {row['id']:row for row in read(ARCHIVE/(panel+'.json.gz'))}


def homography(chart):
    h = np.array(chart['H'])
    return h / h[2, 2]


def linked(name, panel, source):
    if panel == 'sky':
        root = source / 'shared-corner-rows'
        shared = root / 'sky' / name
        frozen = DENSE / name / 'dense-charts.json'
        alternate = root / 'dense-source/sky' / name / 'dense-charts.json'
        dense = alternate if alternate.exists() and digest(alternate) == read(shared/'charts.json')['baseline_chart_sha256'] else frozen
    else:
        shared = source / 'shared/sky' / name
        dense = source / 'shared/dense' / name / 'dense-charts.json'
    return dense, shared / 'charts.json', shared / 'seams.json'


def scan(row, special):
    name, panel = row['id'], row['panel']
    source, connected, _, cases = sources(panel)
    case = next(c for c in cases if c['id'] == name)
    rel = Path('sky') / name / 'charts.json'
    base = T1ROOT / 'occlusion-fragments/original' if special else BASE / panel / 'large-contraction'
    packet = base / 'charts' / rel
    seed = connected / rel
    dense, shared, seams = linked(name, panel, source)
    files = dict(keep=packet, seed=seed, dense=dense, shared=shared, seams=seams)
    docs = {k: read(p) for k,p in files.items()}
    fingerprints = {k:digest(p) for k,p in files.items()}
    image = Path(case['variants']['sky']['image'])
    stamp = digest(image)
    assert stamp == row['source_sha256'] == case['variants']['sky']['source_sha256']
    assert all(d['source_sha256'] == stamp for d in docs.values())
    assert fingerprints['dense'] == docs['shared']['baseline_chart_sha256']
    assert fingerprints['shared'] == docs['keep']['shared_chart_sha256']
    assert all(docs[k]['atlas_sha256']==atlas().provenance['atlas_sha256'] for k in ('keep','seed','dense','shared'))
    assert docs['seams']['surfaces_sha256']==docs['keep']['surfaces_sha256']
    archived = read(base/'assembly'/LOCATION/name/'score.json')
    assert archived['chart_sha256'] == fingerprints['keep']
    assert archived['pack_sha256'] == digest(base/'assembly'/LOCATION/name/'reconstruction.zip')
    if not special:
        receipt = receipts(panel)[name]['exports']['large-contraction']
        assert receipt['pack_sha256']==archived['pack_sha256']
        assert receipt['scoring_reference_sha256']==archived['reference_sha256']
    frame = Path(case['frame'])
    assert all(digest(frame/key)==value for key,value in archived['reference_sha256'].items())
    assert archived['surfaces_sha256']==digest(source/'sky'/name/'source-handoff/surfaces.json')
    charts = {k:{c['id']:c for c in d['charts']} for k,d in docs.items() if 'charts' in d}
    accepted = {s[role+'_id'] for s in docs['seams']['seams'] if s['decision']=='accepted' for role in ('donor','receiver')}
    records = {}
    for cid in sorted(accepted):
        old, shared_chart, final, seed_chart = (charts[k].get(cid) for k in ('dense','shared','keep','seed'))
        status = 'offered'
        if not old or not old['cells']:
            status = 'no earlier measured cells'
        elif not final or final.get('method')!='shared-corner-rows':
            status = 'no surviving corner transform'
        elif not shared_chart or not seed_chart or not all(np.allclose(homography(c), homography(final),rtol=1e-9,atol=1e-9)
                                                           for c in (shared_chart,seed_chart)):
            status = 'different final transform'
        elif np.allclose(homography(old),homography(final),rtol=1e-7,atol=1e-6):
            status = 'same transform'
        records[cid] = dict(status=status, earlier_method=old and old['method'], final_method=final and final['method'],
                            before_cells=len(final['cells']) if final else None)
    result = dict(panel=panel,id=name,dataset=row['dataset'],split=row['split'],charts=records,
                  files={k:str(v) for k,v in files.items()},sha256=fingerprints,source_sha256=stamp,
                  frozen_source_sha256=row['source_sha256'],reference_sha256=archived['reference_sha256'],
                  baseline_pack_sha256=archived['pack_sha256'],base=str(base),special=special,
                  accepted_seams=[{k:s[k] for k in ('donor_id','receiver_id','reason')}
                                  for s in docs['seams']['seams'] if s['decision']=='accepted'],
                  atlas_sha256=docs['keep']['atlas_sha256'])
    return result, case


def regenerate(chart, plane, rgb):
    alpha,(x,y,r,b) = _native_alpha(plane['support'],rgb.shape[1::-1])
    mask = np.zeros(rgb.shape[:2],bool)
    mask[y:b,x:r] = np.asarray(alpha)>0
    points = np.column_stack(np.nonzero(mask)[::-1])+.5
    return preserve_and_rank(chart,resample(chart,np.asarray(chart['H']),mask,rgb,points),rgb,mask)


def execute(job):
    row,special = job
    name,panel = row['id'],row['panel']
    cv2.setNumThreads(1)
    result,case = scan(row,special)
    source = sources(panel)[0]
    base = Path(result['base'])
    folder = OUT/('exact-t1' if special else panel)
    rel = Path('sky')/name/'charts.json'
    targets = [cid for cid,rec in result['charts'].items() if rec['status']=='offered']
    if special:
        targets = ['mask-1']
    packet = read(Path(result['files']['keep']))
    options = {'keep':base/'assembly'}
    packets = {'keep':str(base/'charts'/rel)}
    substitutions = {'keep':[]}
    if targets:
        rgb = np.asarray(Image.open(case['variants']['sky']['image']).convert('RGB'))
        planes = {p['id']:p for p in read(source/'sky'/name/'source-handoff/surfaces.json')['planes']}
        stages = {key:{c['id']:c for c in read(Path(result['files'][key]))['charts']} for key in ('seed','dense')}
        final = {c['id']:c for c in packet['charts']}
        keep = {cid:regenerate(stages['seed'][cid],planes[cid],rgb) for cid in targets}
        assert all(keep[cid]==final[cid] for cid in targets), (panel,name,[cid for cid in targets if keep[cid]!=final[cid]])
        restored = {cid:regenerate(stages['dense'][cid],planes[cid],rgb) for cid in targets}
        for cid in targets:
            result['charts'][cid]['after_cells']=len(restored[cid]['cells'])
        variants = {'noop':keep, **{'restore-'+cid:{cid:restored[cid]} for cid in targets}}
        if len(targets)>1:
            variants['restore-all']=restored
        for arm,replacements in variants.items():
            root = folder / arm
            data = dict(packet,charts=[replacements.get(c['id'],c) for c in packet['charts']])
            save(root/'charts'/rel,data)
            with patch.object(combined,'inputs',sources):
                combined.run((panel,case,root/'charts',root/'assembly'))
            options[arm]=root/'assembly'
            packets[arm]=str(root/'charts'/rel)
            substitutions[arm]=list(replacements)
    summary = summarize([case],options,LOCATION)['rows'][0]
    packs = {arm:read(root/LOCATION/name/'score.json')['pack_sha256'] for arm,root in options.items()}
    if targets:
        assert summary['score']['noop']==summary['score']['keep']
        a,b = [options[k]/LOCATION/name for k in ('keep','noop')]
        assert read(a/'placements.json')==read(b/'placements.json')
        with ZipFile(a/'reconstruction.zip') as first, ZipFile(b/'reconstruction.zip') as second:
            assert set(first.namelist())==set(second.namelist())
            assert all(first.read(n)==second.read(n) for n in first.namelist())
    result.update(summary,packet=packets,pack_sha256=packs,substitutions=substitutions)
    save(folder/'rows'/(name+'.json'),result)
    print(panel,name,targets,summary['paired'],flush=True)


def jobs():
    rows = [r for r in read(FREEZE) if r['active']]
    rows.sort(key=lambda r:(r['id']!=T1,r['panel']=='original'))
    assert len(rows)==84 and len({(r['panel'],r['id']) for r in rows})==84
    return [(r,False) for r in rows]+[(next(r for r in rows if r['id']==T1),True)]


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--resume',action='store_true')
    args=cli.parse_args()
    pending=[(r,s) for r,s in jobs() if not(args.resume and (OUT/('exact-t1' if s else r['panel'])/'rows'/(r['id']+'.json')).exists())]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(execute,pending))
