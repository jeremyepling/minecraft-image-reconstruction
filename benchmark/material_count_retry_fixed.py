"""Remove only the counter's texture-guided retry on the 104 fixed public handoffs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmark import material_feedback_grid as grid, reference_surface_feedback as replay, typed_structures as typed
from engine import eave_pinholes, smooth_atlas_outline, surface_model

OUT = grid.OUT / 'retry-bypass-fixed.json'


def entry(row, faces):
    dims = row['dimensions']['blocks']
    ref = grid.expected(faces,row.get('quad'), include_horizontal=True) if faces else None
    return dict(dimensions=dims, reference=ref, method=row['method'])


def handoff(row):
    # Both visible_placement and counted_ground_corner read visible_placement._charts.
    return row['id'], row.get('quad'), row['counts']


def public_replay(job,counts,folder):
    with patch.object(replay.viewer,'recount',return_value=counts):
        key,_,_ = replay.evaluate(('baseline',job,folder))
    r = typed.read(folder/(key+'--baseline.json'))
    return {k:r[k] for k in ('geometry','scores','pack_sha256')}


def process(control):
    key = control['id']
    job = replay.all_jobs()[key]
    assert control['source_sha256'] == job[1]['files']['rgb.png']
    im,parts,_,labels,ids,_ = replay.prepared(key)
    measured = eave_pinholes.measurement_parts(smooth_atlas_outline.measurement_parts(parts))
    before = replay.viewer.recount(im,measured,None)
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        after = surface_model._count_regions(im,measured,'26.3','masked-repeat',None,folder)
        pairs = grid.pair(parts,im.size,labels,ids)
        changed = []
        tally = Counter()
        for a,b in zip(before['surfaces'],after['surfaces'],strict=True):
            assert a['id']==b['id']
            faces = pairs.get(a['id'])
            old,new = entry(a,faces),entry(b,faces)
            touched = handoff(a)!=handoff(b)
            if touched or old['dimensions']!=new['dimensions']:
                changed.append(dict(surface=a['id'],handoff=touched,ordinary=old,without_retry=new))
            tally['rows']+=1
            if faces:
                tally['paired']+=1
                for name,res in [('ordinary',old),('without_retry',new)]:
                    dims,ref=res['dimensions'],res['reference']
                    if ref is not None:
                        tally[name+'_charted']+=1
                        for i,axis in enumerate(('across','down')):
                            tally[name+'_'+axis+'_returned']+=int(dims[i] is not None)
                            tally[name+'_'+axis+'_correct']+=int(dims[i]==ref[i])
                        tally[name+'_complete']+=int(None not in dims)
                        tally[name+'_exact']+=int(dims==ref)
        old = {k:control['baseline'][k] for k in ('geometry','scores','pack_sha256')}
        rerun = any(r['handoff'] for r in changed)
        if rerun:
            assert public_replay(job,before,folder)==old,key
            new = public_replay(job,after,folder)
        else:
            new=old
    return dict(id=key,split=control['split'],source_sha256=control['source_sha256'],rerun=rerun,tally=dict(tally),changed=changed,ordinary=old,without_retry=new)


def verify_builds(directory):
    """Check ordinary bypass builds against the paired public handoff."""
    from benchmark.mask_blocks import decoded_pack

    record = typed.read(OUT)
    cases = {row['id']:row for row in record['cases']}
    controls = {row['id']:row for row in typed.read(replay.ROOT/'fixed-confirmation.json')['cases']}
    result = {}
    for key in record['summary']['changed_packs']:
        row = cases[key]
        folder = directory/key
        assert typed.read(folder/'build-manifest.json')['image_sha256'] == row['source_sha256']
        assert controls[key]['reproduced_pack']
        actual = decoded_pack(folder/'reconstruction.zip')
        payload, _ = typed.typed_pack({p:v.removeprefix('minecraft:') for p,v in actual.items()})
        canonical = sha256(payload).hexdigest()
        base = controls[key]['released_pack_sha256']
        job = replay.all_jobs()[key]
        labels,ids = replay.visible_faces(*job[:2])
        _,_,matrix,_ = replay.ref_inputs(job)
        truth = {tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
        optional = replay.horizontal_slivers(labels,ids)
        coords = typed.aligned_sites(list(actual),matrix,(0,0,0))
        shift = replay.best_translation(coords,truth)[0]
        points = typed.aligned_sites(list(actual),matrix,shift)
        placed = dict(zip(points,actual.values(),strict=True))
        sources = {tuple(b[k] for k in 'xyz'):b['material_source'] for b in typed.read(folder/'placements.json')['blocks']}
        assigned = {p for p,c in zip(points,actual,strict=True) if sources[c]!='fallback'}
        result[key] = dict(source_sha256=row['source_sha256'],ordinary_pack_sha256=base,
                           bypass_canonical_sha256=canonical,matched_public_bypass=canonical==row['without_retry']['pack_sha256'],
                           geometry=typed.geometry_score(placed,truth,optional),
                           scores=typed.score(placed,truth,optional,assigned))
    typed.compact(grid.OUT/'retry-bypass-ordinary.json',result)
    print(json.dumps(result))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--verify-builds',type=Path)
    args=parser.parse_args()
    if args.verify_builds:
        verify_builds(args.verify_builds)
        return
    fixed=replay.ROOT/'fixed-confirmation.json'
    controls=typed.read(fixed)['cases']
    assert len(controls)==len({r['id'] for r in controls})==104
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(process,controls))
    total=Counter()
    for r in rows:
        total.update(r['tally'])
    summary=dict(rows=104,rerun=sum(r['rerun'] for r in rows),changed_packs=[r['id'] for r in rows if r['ordinary']['pack_sha256']!=r['without_retry']['pack_sha256']],tally=dict(total))
    for mode in ('ordinary','without_retry'):
        summary[mode]={k:typed.pooled_scores([r[mode][k] for r in rows]) for k in ('geometry','scores')}
    typed.compact(OUT,dict(control_sha256=sha256(fixed.read_bytes()).hexdigest(),summary=summary,cases=rows))
    print(json.dumps(summary))


if __name__=='__main__':
    main()
