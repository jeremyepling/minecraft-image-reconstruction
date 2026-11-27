"""Replay the ordinary Java 26.3 entry with a count-only intervention at placement.

Invoke via Invoke-WallRecovery.ps1 -Python benchmark/CountOraclePython.ps1.
The first pass runs unmodified. Repeated image-conditioned model calls reuse
their exact prior inputs; all stages downstream of the intervention rerun.
"""
from contextlib import ExitStack
from copy import deepcopy
from functools import wraps
from hashlib import sha256
import os
import pickle
from pathlib import Path
import sys
from unittest.mock import patch

from benchmark import material_feedback_grid as grid
from benchmark.relative_geometry import best_translation
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import mlsd_tiny, planarity_entry, recovery_evidence, surface_fields, surface_model, visible_placement
from engine.surface_dimensions import dimensions
import reconstruct


def cached(fn):
    cache={}
    @wraps(fn)
    def call(*args,**kwargs):
        key=sha256(pickle.dumps((args,kwargs), protocol=5)).digest()
        if key not in cache:
            cache[key]=fn(*args,**kwargs)
        return deepcopy(cache[key])
    return call


def main():
    cli=sys.argv[1:]
    source=Path(cli[2])
    out=Path(cli[cli.index('--out')+1])
    out.mkdir(exist_ok=True,parents=True)
    fingerprint=digest(source)
    job=next(j for j in grid.typed.SUITE.jobs(grid.typed.SUITE.framed.EVALUATION) if j[1]['files']['rgb.png']==fingerprint)
    key=job[1]['id']
    original=visible_placement.place
    exports={}
    changed={}
    artifacts={}
    trace={}
    bypass_only=os.environ.get('MCIR_COUNT_BYPASS_ONLY')=='1'
    prior=read(out/'result.json') if bypass_only else None
    if bypass_only and (prior['source_hash']!=fingerprint or not prior['matched_release'] or
                        prior['cases']['control']['pack_sha256']!=digest(out/'control/reconstruction.zip')):
        raise ValueError('Bypass needs the previously verified paired control')
    def intervene(image,version,parts,*,counts,**kw):
        # Some later local repairs invoke place() with their own counterfactual
        # masks. Only the primary call within build() receives the intervention.
        if sys._getframe(1).f_code is not surface_model.build.__code__:
            return original(image,version,parts,counts=counts,**kw)
        byid={p['id']:p for p in parts}
        rows=counts['surfaces']
        if mode=='control':
            trace['parts']=deepcopy(parts)
            trace['counts']=deepcopy(counts)
            trace['camera']=deepcopy(kw['camera'])
        elif not bypass_only and (parts!=trace['parts'] or rows!=trace['counts']['surfaces'] or kw['camera']!=trace['camera']):
            raise ValueError('Primary placement inputs differ between the paired builds')
        if mode=='disabled':
            # Bypass only the raw spacings handed to this primary call. Later
            # postprocessing sees its unaltered counts, phase, chart and masks.
            counts=deepcopy(counts)
            for row in counts['surfaces']:
                measured=row['counts']
                if row.get('quad') is not None and any(v is not None for v in measured):
                    changed[mode].append([row['id'],measured,[None,None]])
                    row['counts']=[None,None]
            return original(image,version,parts,counts=counts,**kw)
        if mode=='oracle':
            matched=grid.pair(parts,image.size,labels,rays)
        if mode=='feedback':
            hints=grid.propose(image,parts,counts,kw['camera'],site)
        for r in rows:
            name=r['id'];quad=r.get('quad')
            if name not in byid or quad is None:
                continue
            dims=dimensions(byid[name],quad,image.size,r['counts'])['blocks']
            target=None
            if mode=='oracle' and name in matched:
                target=grid.expected(matched[name],quad)
            if mode=='feedback' and name in hints and hints[name]['corners']:
                target=hints[name]['dims']
            if target is None or target==dims:
                continue
            extent=dimensions(byid[name],quad,image.size,[1.,1.])['spans']
            if any(e is None or e<=0 for e in extent):
                continue
            r['counts']=[n/e if n!=old else c for n,e,old,c in zip(target,extent,dims,r['counts'])]
            r['dimensions']=dimensions(byid[name],quad,image.size,r['counts'])
            if r['dimensions']['blocks']!=target:
                raise ValueError('Requested span did not reach the placer')
            changed[mode].append([name,dims,target])
        return original(image,version,parts,counts=counts,**kw)
    with ExitStack() as stack:
        for mod,attr in ((surface_fields,'infer'),(recovery_evidence,'infer'),(mlsd_tiny,'detect'),(planarity_entry,'infer')):
            stack.enter_context(patch.object(mod,attr,cached(getattr(mod,attr))))
        stack.enter_context(patch.object(visible_placement,'place',intervene))
        modes=('disabled',) if bypass_only else ('control','sham','feedback','oracle','disabled')
        for mode in modes:
            if mode=='feedback':
                blocks=read(out/'control/placements.json')['blocks']
                site={'rows':[[b['x'],b['y'],b['z'],'minecraft:'+b['block_id'],b['material_source']] for b in blocks]}
            if mode=='oracle' or bypass_only:
                labels,rays=grid.surfaces.visible_faces(job[0],job[1])
            changed[mode]=[]
            cfg=cli.copy()
            cfg[cfg.index('--out')+1]=str(out/mode)
            with patch.object(sys,'argv',cfg):
                reconstruct.main()
            artifact=out/mode/'reconstruction.zip'
            exports[mode]=decoded_pack(artifact)
            artifacts[mode]=digest(artifact)
            if mode=='sham' and exports[mode]!=exports['control']:
                raise ValueError('No-op replay changed placed geometry or IDs')
            if mode!='control' and any(digest(out/mode/name)!=digest(out/'control'/name)
                                       for name in ('surfaces.json','counts.json')):
                raise ValueError('Public surface or count output differs upstream of intervention')
    _,_,registration,_=ref_inputs(job)
    truth={tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
    optional=horizontal_slivers(labels,rays)
    frozen=grid.typed_inputs()[key]
    expected={tuple(r[:3]):r[3] for r in frozen['rows']}
    if decoded_pack(out/'control/reconstruction.zip')!=expected:
        raise ValueError(f'Replay of {key} differs from the ordinary released coordinates and IDs')
    cases=prior['cases'] if bypass_only else {}
    for mode,blocks in exports.items():
        points=grid.typed.aligned_sites(list(blocks),registration,(0,0,0))
        shift,_=best_translation(points,truth)
        world=grid.typed.aligned_sites(list(blocks),registration,shift)
        pred=dict(zip(world,blocks.values(),strict=True))
        sources={tuple(b[k] for k in 'xyz'):b['material_source'] for b in read(out/mode/'placements.json')['blocks']}
        assigned={w for c,w in zip(blocks,world,strict=True) if sources[c]!='fallback'}
        cases[mode]=dict(geometry=grid.typed.geometry_score(pred,truth,optional),
                         typed=grid.typed.score(pred,truth,optional,assigned),
                         translation=shift,pack_sha256=artifacts[mode],changes=changed[mode])
    save(out/'result.json',dict(id=key,source_hash=fingerprint,matched_release=True,cases=cases,
         masks_sha256=digest(out/'control/surfaces.json'),counts_sha256=digest(out/'control/counts.json')))
    print(out/'result.json',flush=True)


if __name__=='__main__':
    main()
