"""Replay ordinary exported material inventories against point and finite-pixel rerankers."""
import argparse
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from benchmark.expanded_baseline import ROOT,WORKTREES,SOURCE,read,save,digest
from benchmark.expanded_fixed_handoffs import _export_atlas
from benchmark.expanded_scores import load_reference
from benchmark.typed_structures import aligned_sites,score,typed_pack,pooled_scores,geometry_score
from benchmark.mask_blocks import decoded_pack
from benchmark.view_conditioned_atlas_match import rerank

BASE = WORKTREES/'full-expanded-controls/runs/expanded-baseline-01'
FIXED = WORKTREES/'typed-structure-scores/runs/typed-structures'
OUT = ROOT/'runs/view-conditioned-atlas'
REVIEW = ROOT/'benchmark/reviews/view-conditioned-atlas-01'

@lru_cache(maxsize=1)
def entries():
    return {(r['dataset'],r['id']):r for r in read(BASE/'source_revision.json')['frames']}

def inputs(row):
    key,ds = row['id'],row['dataset']
    if ds == 'fixed-104':
        name = key+'.json'
        inp = read(FIXED/'inputs'/name)
        with ZipFile(ROOT/'benchmark/reviews/typed-structures-01/typed-exports.zip') as archive:
            pack = archive.read(key+'/reconstruction.zip')
        estimates = read(FIXED/'all-v5'/name)['methods']['auto']
        local = list(map(tuple,inp['cells']))
        evidence = {p:e for p,(n,e) in zip(local,estimates,strict=True)}
        return Path(inp['rgb']),inp['camera'],pack,evidence
    folder = BASE/ds/key
    info = read(folder/'handoffs/material-input.json.gz')
    evidence = {(r['x'],r['y'],r['z']):r['material_source'] for r in read(folder/'placements.json')['blocks']}
    return Path(entries()[ds,key]['frame'])/'rgb.png',info['camera'],(folder/'reconstruction.zip').read_bytes(),evidence

def grading(row,local):
    ds,key = row['dataset'],row['id']
    if ds == 'fixed-104':
        r = read(FIXED/'reference'/(key+'.json'))
        cells = list(map(tuple,read(FIXED/'inputs'/(key+'.json'))['cells']))
        return dict(zip(cells,map(tuple,r['aligned']),strict=True)), {tuple(v[:3]):v[3] for v in r['truth']},set(map(tuple,r['optional']))
    ref = load_reference(entries()[ds,key])
    matrix = np.array(row['registration']['transform'])
    return dict(zip(local,aligned_sites(list(local),matrix,row['translation']),strict=True)),ref.truth,ref.optional

def run(row):
    ds,key = row['dataset'],row['id']
    path = OUT/'cases'/ds/(key+'.json')
    if path.exists():
        return read(path)
    rgb,camera,pack,evidence = inputs(row)
    assert digest(rgb)==row['source_sha256'] and sha256(pack).hexdigest()==row['pack_sha256']
    local = decoded_pack(BytesIO(pack))
    assert local.keys()==evidence.keys()
    ids = {p:n.removeprefix('minecraft:') for p,n in local.items()}
    with Image.open(rgb) as im:
        static,view,details = rerank(im,camera,ids,_export_atlas())
    where,truth,optional = grading(row,local)
    assigned = {where[p] for p,e in evidence.items() if e!='fallback'}
    def grade(names):
        return score({where[p]:n for p,n in names.items()},truth,optional,assigned)
    baseline = grade(local)
    assert baseline==row['scores'],(ds,key,baseline,row['scores'])
    assert geometry_score({where[p]:v for p,v in local.items()},truth,optional)==row['geometry']
    arms = {}
    for name,alternative in [('point',static),('source',view)]:
        payload,decoded = typed_pack(alternative)
        assert decoded.keys()==local.keys()
        arms[name] = grade(decoded)
        if name=='source':
            out = OUT/'packs'/ds/(key+'.zip')
            out.parent.mkdir(exist_ok=True,parents=True)
            out.write_bytes(payload)
            pack_hash = sha256(payload).hexdigest()
    oracle = {p:local[p] if local[p]==truth.get(where[p]) else decoded[p] for p in local}
    arms['oracle'] = grade(oracle)
    for rec in details:
        position = where[tuple(rec['xyz'])]
        rec['truth'] = truth.get(position)
    save(OUT/'details'/ds/(key+'.json.gz'),details)
    result = dict(dataset=ds,id=key,split=row['split'],layout=row['layout'],source_sha256=row['source_sha256'],
                  baseline_pack_sha256=row['pack_sha256'],pack_sha256=pack_hash,cells_sha256=sha256(str(sorted(ids)).encode()).hexdigest(),
                  exported=len(ids),evaluated_cells=len(details),changed=sum(view[p]!=ids[p] for p in ids),
                  changed_from_point=sum(view[p]!=static[p] for p in ids),baseline=baseline,geometry=row['geometry'],**arms)
    save(path,result)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    rows=read(BASE/'scores/cases.json')
    if args.limit:
        rows=rows[:args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        futures={pool.submit(run,r):r for r in rows}
        for n,f in enumerate(as_completed(futures)):
            try:
                f.result()
            except Exception:
                print('ERROR',futures[f]['id'],flush=True)
                raise
            if n%10==0:
                print(n+1,'/',len(rows),flush=True)
    cases=[read(OUT/'cases'/r['dataset']/(r['id']+'.json')) for r in rows]
    save(REVIEW/'cases.json',cases)
    summary = {}
    for ds in sorted({r['dataset'] for r in rows}):
        selected=[r for r in cases if r['dataset']==ds]
        summary[ds]=dict(views=len(selected),changed=sum(r['changed'] for r in selected),
                         effects_of_integration=sum(r['changed_from_point'] for r in selected),
                         **{arm:pooled_scores([r[arm] for r in selected]) for arm in ('baseline','point','source','oracle')})
    save(REVIEW/'summary.json',summary)
    print(summary)
if __name__=='__main__':
    main()
