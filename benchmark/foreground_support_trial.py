"""Compare source-only foreground vetoes to visible reference surfaces and placed blocks."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from benchmark import current_placement_viewer as v, smooth_joint_placement as joint
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.thin_face_placement_run import ref_inputs
from benchmark.texture_camera_eval import read, save
from engine.pillar_seams import native_mask
from engine.plane_candidates import foreground
from engine.projected_appearance import Scene


def backgrounds(image):
    low = tuple(max(1, x//3) for x in image.size)
    rgb = np.asarray(image.resize(low, Image.Resampling.BOX)).astype('int16')
    width = max(1,rgb.shape[1]//32)
    def margin(samples):
        refs = np.percentile(samples,[10,50,90],axis=1).astype('int16')
        return np.min([np.abs(rgb-r[:,None]).max(2) for r in refs],axis=0)
    distance = np.maximum(margin(rgb[:,:width]),margin(rgb[:,-width:]))
    # Background should be reachable without crossing dissimilar pixels.
    candidate = np.pad(distance <= 4, 1, constant_values=True)
    filled = Image.fromarray(candidate.astype('uint8')).copy()
    ImageDraw.floodfill(filled,(0,0),2)
    exterior = Image.fromarray(np.uint8((np.asarray(filled)[1:-1,1:-1]==2)*255))
    _, core = foreground(image)
    outside = Image.fromarray(np.uint8(255*~core)).filter(ImageFilter.MinFilter(5))
    def up(field):
        return np.asarray(field.resize(image.size,Image.Resampling.NEAREST))>0
    result={'largest':up(outside),'connected':up(exterior),
            'guarded':up(exterior.filter(ImageFilter.MinFilter(7)))}
    # A slowly changing floor can continue from either image edge even where
    # its local shade disagrees with both same-height margins.
    edge=np.zeros(low[::-1],dtype=np.int16)
    for axis in (0,1):
        delta=np.max(np.abs(np.diff(rgb,axis=axis)),axis=2)
        if axis==0:
            edge[:-1]=np.maximum(edge[:-1],delta)
            edge[1:]=np.maximum(edge[1:],delta)
        else:
            edge[:,:-1]=np.maximum(edge[:,:-1],delta)
            edge[:,1:]=np.maximum(edge[:,1:],delta)
    for threshold in (3,6,10):
        free=Image.fromarray(np.uint8(np.pad(edge<=threshold,1,constant_values=True))).copy()
        ImageDraw.floodfill(free,(0,0),2)
        field=Image.fromarray(np.uint8(255*(np.asarray(free)[1:-1,1:-1]==2)))
        result['edge'+str(threshold)]=up(field)
    return result


@lru_cache(maxsize=1)
def controls():
    folders=['smooth-local-placement-01','smooth-joint-placement-01','window-wall-01','counted-ground-corner-01']
    _,history,payloads=v.read_releases([v.RELEASE,*[v.ROOT/p for p in folders]])
    expected={r['id']:r for r in read(v.ROOT/'facade-inset-round-01/combined.json')['rows']}
    baseline={r['id']:r for r in read(v.ROOT/'counted-ground-corner-01/cases.json')}
    return expected,baseline,payloads


def audit(job):
    split,frame,*_=job
    key=frame['id']
    expected,baseline,payloads=controls()
    if key in expected:
        image,parts,_,camera=v.latest_stages(job,expected[key])
    else:
        from engine import inset_returns,eave_pinholes,jagged_texture_join,smooth_atlas_outline
        image,parts,_,camera=v.suite.ordinary_inputs(*job)
        parts,rim,phase,outlines,_=inset_returns.detect(image,parts,'26.3',camera=camera)
        parts,_=eave_pinholes.refine(image,parts)
        if not(rim or outlines or phase):
            parts,_,_=jagged_texture_join.refine(image,parts,'26.3',camera=camera)
        parts,_=smooth_atlas_outline.refine(image,parts,'26.3',atlas_loader=v.calibrated_atlas)
    bg=backgrounds(image)
    # Fix proposals from RGB and model masks before opening scoring labels.
    masks={p['id']:native_mask(p,image.size) for p in parts}
    trim={}
    for name,out in bg.items():
        trim[name]={}
        for sid,m in masks.items():
            if m.any() and np.count_nonzero(m & out) >= .65*np.count_nonzero(m):
                trim[name][sid]=m & out
    truth=v.suite.visible_map(split,frame)>0
    result={}
    for name,proposed in trim.items():
        events=[]
        for sid,m in proposed.items():
            events.append(dict(mask=sid,part=int(masks[sid].sum()),background=int((m&~truth).sum()),structure=int((m&truth).sum())))
        result[name]=dict(events=events)
    if not camera or camera.get('P') is None:
        return dict(id=key,split=split,sha=frame['files']['rgb.png'],methods=result)
    b=baseline[key]
    if key in payloads:
        pred=decoded_pack(BytesIO(payloads[key]))
        assert sha256(payloads[key]).hexdigest()==b.get('pack_sha256',b['previous_pack_sha256'])
    else:
        h,pred=joint.old_pack(job)
        assert h==b.get('pack_sha256',b['previous_pack_sha256'])
    scene=Scene(image,parts,camera,pred)
    cells={name:set() for name in bg}
    for name,out in bg.items():
        samples=out[::2,::2].ravel()
        for i,pixels in scene.owned.items():
            if len(pixels)>=4 and np.all(samples[pixels]):
                cells[name].add(scene.cells[i])
    if any(cells.values()):
        target,_,transform,_=ref_inputs(job)
        shift=b['result']['translation']
        for name,proposed in cells.items():
            world=v.suite.framed.spacing.shift(registered(proposed,transform),shift)
            result[name].update(correct_vetoed=len(world&target),extras_vetoed=len(world-target))
    return dict(id=key,split=split,sha=frame['files']['rgb.png'],methods=result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=('focus','families','all'),default='families')
    parser.add_argument('--workers',type=int,default=5)
    a=parser.parse_args()
    jobs=list(v.suite.jobs([*v.suite.framed.SIZES,'shelter']))
    if a.group=='focus':
        jobs=[j for j in jobs if j[1]['id']=='legacy-light_gray_concrete-holdout-workshop-plaster-high-right']
    elif a.group=='families':
        jobs=[j for j in jobs if j[0] in v.suite.framed.EVALUATION and any(x in j[1]['id'] for x in ('workshop','galleries','shopfront'))]
    with ProcessPoolExecutor(a.workers) as pool:
        rows=list(pool.map(audit,jobs))
    path=v.suite.ROOT/'runs/workshop-foreground-support'
    path.mkdir(exist_ok=True,parents=True)
    save(path/(a.group+'.json'),rows)
    for group in ('workshop','galleries','shopfront','all'):
        for name in ('largest','connected','guarded','edge3','edge6','edge10'):
            stats=Counter()
            for r in rows:
                if group!='all' and group not in r['id']:continue
                m=r['methods'][name]
                stats.update(frames=1,hits=len(m['events']),correct_vetoed=m.get('correct_vetoed',0),extras_vetoed=m.get('extras_vetoed',0))
                for e in m['events']: stats.update(background=e['background'],structure=e['structure'])
            print(group,name,dict(stats))
if __name__=='__main__':main()
