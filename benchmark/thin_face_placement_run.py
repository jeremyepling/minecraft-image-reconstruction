"""Replay counted-face and thin neighbor proposals against the shipped 204-view packs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark import framed_depth_eval as framed
from benchmark.inset_default_verification import local_release
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement import propose
from engine import inset_returns, surface_model, texture_guided_count
from engine.pillar_seams import native_mask, orientation
from engine.partial_atlas_face import verify
from engine.atlas_count import load_atlas
from engine.visible_raster import SAMPLE_STRIDE
from engine.visible_placement import _sample_face
from engine.thin_atlas_faces import _mips, _match

HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'runs/thin-face'
REPORT=HERE/'benchmark/reviews/thin-face-placement-01'
COHORTS=dict(development=['train'],fixed=framed.EVALUATION,alternate=['extra'],new=['new'],shelter=['shelter'])


@lru_cache(maxsize=1)
def atlas():
    return load_atlas(suite.ATLAS,'26.3')


@lru_cache(maxsize=1)
def maps():
    return _mips(atlas())


def current_inputs(job):
    split,frame,_,control=job
    key=frame['id']
    im,old,counts,camera=suite.ordinary_inputs(*job)
    row,z=local_release()
    row=row[key]
    assert row['rgb_sha256']==frame['files']['rgb.png']
    method=row['methods'].get('clear_front')
    if method or split=='shelter':
        if method:
            data=z.read('clear_front/'+split+'/'+key+'/reconstruction.zip')
        else:
            with ZipFile(suite.ARCHIVE) as archive:
                data=archive.read('production/transfer/'+key+'/reconstruction.zip')
        assert sha256(data).hexdigest()==(method or control)['pack_sha256']
        original=decoded_pack(BytesIO(data))
    else:
        original=decoded_pack(checked(suite.CONTROL/split/key/'reconstruction.zip',control['pack_sha256']))
    assert len(original)==sum((method or row['baseline'])[s] for s in ('correct','extras'))
    updated,strict,phase,outline,_=inset_returns.detect(im,old,'26.3',camera=camera)
    if strict or outline:
        old_masks={p['id']:p.get('support') for p in old}
        changed=[p for p in updated if p.get('support')!=old_masks.get(p['id'])]
        selected_atlas=atlas() if 'atlas_phase' in counts else None
        with TemporaryDirectory() as tmp:
            new=surface_model._count_regions(im,changed,'26.3',texture_guided_count.METHOD,selected_atlas,Path(tmp))
        rows={r['id']:r for r in counts['surfaces']}
        rows.update((r['id'],r) for r in new['surfaces'])
        counts=dict(surfaces=[rows[p['id']] for p in updated])
    return im,updated,counts,camera,set(original),method or row['baseline'],(method or control)['pack_sha256']


def verify_texture(im,parts,camera,rows):
    if not rows:
        return
    P,_=orientation(camera,im.size)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    rgb=np.asarray(im.convert('RGB'))
    masks={p['id']:p for p in parts}
    @lru_cache(maxsize=None)
    def get_mask(key):
        mask=native_mask(masks[key],im.size)
        return mask[SAMPLE_STRIDE//2::SAMPLE_STRIDE,SAMPLE_STRIDE//2::SAMPLE_STRIDE],mask
    @lru_cache(maxsize=None)
    def score(key,cell,axis):
        mask,native=get_mask(key)
        return bool(verify(rgb,atlas(),mask,P,cell,axis,int(eye[axis]>cell[axis]+1)*2-1,
                           native_mask=native,threshold=.55,minimum=32))
    @lru_cache(maxsize=None)
    def partial(key,cell,axis):
        level=cell[axis]+int(eye[axis]>cell[axis]+1)
        native=P.copy(); native[:2]*=SAMPLE_STRIDE
        hit=_sample_face(native,cell,axis,1 if level>cell[axis] else -1,im.size[::-1])
        if hit is None:return None
        box,poly,*_=hit
        y,x=np.nonzero(poly & get_mask(key)[1][box])
        if len(y)<8:return None
        idx=np.linspace(0,len(y)-1,min(240,len(y)),dtype=int)
        y,x=y[idx]+box[0].start,x[idx]+box[1].start
        rays=np.column_stack((x+.5,y+.5,np.ones(len(x)))) @ np.linalg.inv(P[:,:3]).T
        world=eye+(level-eye[axis])*rays/rays[:,axis,None]
        free=[i for i in range(3) if i!=axis]
        return _match(np.floor(16*world[:,free]).astype(int),rgb[y,x],atlas(),maps())
    for row in rows:
        for face in ('host','side'):
            row[face+'_atlas']=score(row[face],row['cell'],row[face+'_axis'])
            if not row['existing']:
                m=partial(row[face],row['cell'],row[face+'_axis'])
                row[face+'_partial']=({k:m[k] for k in ('split','phase','color','texels','sprite')} if m else None)


def ref_inputs(job):
    split,frame,_,_=job
    root=suite.SHELTER if split=='shelter' else framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
    inputs=suite.SHELTER if split=='shelter' else framed.NATIVE_INPUTS.get(split)
    ref='old' if split in ('shelter','extra','new') else split
    return framed.references(ref,frame,inputs,framed.TRAINING_DATA,framed.DEPTH,
                             dict(native=framed.native.BUILDS,old=root,fresh=root))


def case(task):
    job,out=task
    split,f,_,ctl=job
    im,parts,counts,camera,cells,base,pack=current_inputs(job)
    rows=propose(im,parts,counts,camera,cells)
    verify_texture(im,parts,camera,rows)
    # Only the evaluator below reads recorded poses and visible world geometry.
    if rows:
        truth,occupied,T,_=ref_inputs(job)
        shift=np.asarray(base.get('translation',ctl['translation']))
        transformed=registered(cells,T)
        assert sum(tuple(p+shift) in truth for p in transformed)==base['correct'],f['id']
        for r in rows:
            world=next(iter(registered([r['cell']],T)))
            point=tuple(world+shift)
            r.update(reference=point in truth,hidden=point in occupied and point not in truth)
            if r.get('conflict'):
                rival=next(iter(registered([r['conflict']['cell']],T)))
                r['conflict']['reference']=tuple(rival+shift) in truth
                r['conflict']['hidden']=tuple(rival+shift) in occupied and tuple(rival+shift) not in truth
    result=dict(id=f['id'],split=split,rgb=f['files']['rgb.png'],pack=pack,
                baseline={k:base[k] for k in ('correct','missed','extras','hidden_extras')},rows=rows)
    save(out/(f['id']+'.json'),result)
    return f['id'],len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',default=['train'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    jobs=[(j,args.out) for j in suite.jobs(args.splits)]
    with ProcessPoolExecutor(args.workers) as pool:
        result=list(pool.map(case,jobs))
    print(dict(frames=len(result),proposals=sum(n for _,n in result),active=sum(n>0 for _,n in result)))
