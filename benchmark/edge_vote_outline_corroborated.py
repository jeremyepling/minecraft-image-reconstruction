"""Corroborate the projected row with source-only two-face fits and exact cube-ray ownership."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.edge_vote_outline import adjustments,revised_parts
from benchmark.edge_vote_outline_eval import OUT,baseline_pack,changes,geometry,recount
from benchmark.texture_camera_eval import digest,read,save
from engine import inset_returns,surface_model,texture_guided_count
from engine.pillar_seams import native_mask,orientation
from engine.surface_boundary_refine import _retain_topology
from engine.surface_perimeter import with_outline

WITNESSES=OUT/'two-face-witnesses.zip'


def witnesses(path, key, image_hash, pack_hash):
    if path.suffix=='.zip':
        with ZipFile(path) as source:
            frame=json.loads(source.read(key+'.json'))
    else:
        frame=read(path/(key+'.json'))
    assert frame['id']==key and frame['rgb']==image_hash and frame['pack']==pack_hash
    # Drop scoring labels at the boundary. Only the measured two-face observations enter inference.
    return {(r['host'],r['side'],tuple(r['cell'])) for r in frame['rows']
            if r['existing'] and r['count_axes']==2 and r['pitch_error']<=.20
            and r['host_visible']>=.60 and r['side_visible']>=.60
            and r['host_atlas'] and r['side_atlas']}


def first_hits(flat,size,camera,cells):
    """Native-pixel slab intersection with every previously placed cube, including occluders."""
    if not len(flat):
        return [],np.empty(0,int)
    P,_=orientation(camera,size)
    inv=np.linalg.inv(P[:,:3])
    eye=-inv@P[:,3]
    coordinates=np.array(sorted(cells),int)
    low=coordinates-eye
    high=coordinates+1-eye
    result=[]; axes=[]
    for batch in np.array_split(flat,max(1,(len(flat)+127)//128)):
        y,x=np.divmod(batch,size[0])
        ray=np.stack((x+.5,y+.5,np.ones(len(x))),axis=1)@inv.T
        ray=np.where(abs(ray)<1.e-12,1.e-12,ray)
        t0=low[None,:,:]/ray[:,None,:]
        t1=high[None,:,:]/ray[:,None,:]
        enter=np.minimum(t0,t1)
        near=enter.max(axis=2)
        far=np.maximum(t0,t1).min(axis=2)
        near=np.where((near>0)&(far>=near),near,np.inf)
        ix=near.argmin(axis=1)
        for j,idx in enumerate(ix):
            if np.isfinite(near[j,idx]):
                result.append(tuple(coordinates[idx]));axes.append(int(enter[j,idx].argmax()))
            else:
                result.append(None);axes.append(-1)
    return result,np.array(axes)


def guarded_pixels(image,parts,counts,camera,cells,witness):
    owner,runs,events=adjustments(image,parts,counts,camera,cells)
    new=owner.copy()
    veto=np.zeros(owner.shape,bool)
    logs=[]
    for e in events:
        if e['cells']<4 or e['margin']<4 or e['strength']<10 or abs(e['offset'])>1 or not e['clear'].any():
            continue
        host=parts[e['host']-1]['id'];side=parts[e['side']-1]['id']
        members=set(e['members'])
        if any((host,side,c) not in witness for c in members):continue
        inside=e['clear']
        ys,xs=np.unravel_index(e['flat'][inside],owner.shape)
        flat=e['flat'][inside]
        target=e['after'][inside]
        # Independently reject a row if any contested pixel or one of its four
        # native neighbors belongs to an occluding cube outside this very row.
        rays=np.column_stack((flat,np.ravel_multi_index((ys,xs+1),owner.shape,mode='clip'),
            np.ravel_multi_index((ys,xs-1),owner.shape,mode='clip'),
            np.ravel_multi_index((ys+1,xs),owner.shape,mode='clip'),
            np.ravel_multi_index((ys-1,xs),owner.shape,mode='clip')))
        hit,axes=first_hits(rays.ravel(),image.size,camera,cells)
        if any(h not in members for h in hit):continue
        required=np.where(target==e['host'],e['host_axis'],e['side_axis'])
        if np.any(axes.reshape(-1,5)[:,0]!=required):continue
        old=new.flat[flat]
        veto.flat[flat]|=(old!=owner.flat[flat])&(old!=target)
        new.flat[flat]=target
        logs.append(dict(host=host,side=side,axis=[e['host_axis'],e['side_axis']],cells=len(members),
                         xyxy=e['xyxy'],offset=e['offset'],pixels=len(flat)))
    new[veto]=owner[veto]
    if logs:new=_retain_topology(owner,new,dict(radius=3))
    return owner,new,logs


def observed_readings(counts):
    return {row['id']:dict(dimensions=row['dimensions']['blocks'],pitch=row['counts'],phase=row['phase'],
        footprint=[(f['cell'],f['quad']) for k in ('visible_cells','under_five_px_faces') for f in row.get(k,[])],
        material=[f.get('material',{}).get('block_id') for k in ('visible_cells','under_five_px_faces') for f in row.get(k,[])])
        for row in counts['surfaces']}


@lru_cache(maxsize=1)
def material_atlas():
    from engine.atlas_count import load_atlas
    return load_atlas(suite.ATLAS,'26.3')


def verify(work):
    job,run_pack=work
    split,frame,prior,control=job
    image,orig,counts,camera=suite.ordinary_inputs(*job)
    parts,strict,phase,extra,_=inset_returns.detect(image,orig,'26.3',camera=camera)
    atlas=material_atlas() if 'atlas_phase' in counts else None
    if parts is not orig:
        parts=[with_outline(p) for p in parts]
        counts=recount(image,parts,atlas)
        surface_model._inset_layout_counts(image,'26.3',[with_outline(p) for p in orig],counts,
            {event['host'] for event in strict+extra},texture_guided_count.METHOD,atlas)
    pack,cells=baseline_pack(job)
    w=witnesses(WITNESSES,frame['id'],frame['files']['rgb.png'],pack)
    row=dict(id=frame['id'],split=split,rgb=frame['files']['rgb.png'],pack=pack,witnesses=len(w))
    if len(w)<4:return row
    owner,new,ops=guarded_pixels(image,parts,counts,camera,cells,w)
    changed=np.flatnonzero(owner!=new)
    if not len(changed):return row
    row['ops']=ops
    proposal=revised_parts(parts,owner,new)
    active=[i for i,(a,b) in enumerate(zip(parts,proposal)) if a is not b]
    readings={}
    for name,aa in [('ordinary',atlas),('atlas',material_atlas())]:
        before=observed_readings(recount(image,[parts[i] for i in active],aa))
        after=observed_readings(recount(image,[proposal[i] for i in active],aa))
        diff={ident:{k:[b[k],after[ident][k]] for k in b if b[k]!=after[ident][k]} for ident,b in before.items()}
        readings[name]={k:v for k,v in diff.items() if v}
    row['reading_changes']=readings
    def tiny_count_drift(key,pair):
        return key=='pitch' and all((a is None and b is None) or
               (a is not None and b is not None and abs(a-b)<=.01) for a,b in zip(*pair))
    # Fractional periods across a whole chart are an estimate. Preserve all reported integer dimensions,
    # every native grid offset and every sampled tile/material. Log even the allowed sub-.01 period drift.
    row['accepted']=all(tiny_count_drift(key,pair) for changes in readings.values()
                        for columns in changes.values() for key,pair in columns.items())
    truth=suite.visible_map(split,frame)
    row['pixel_grade']=changes(parts,proposal,truth,(owner,new,changed))
    if row['accepted'] and run_pack:
        row['geometry']=geometry(job,image,orig,parts,proposal,counts,camera,cells,strict,extra)
    return row


def record(task):
    job,pack,path=task
    row=verify((job,pack))
    save(path/(row['id']+'.json'),row)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',default=list(suite.framed.SIZES)+['shelter'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=5)
    p.add_argument('--geometry',action='store_true')
    a=p.parse_args()
    location=a.out.with_suffix('');location.mkdir(parents=True,exist_ok=True)
    work=[(job,a.geometry,location) for job in suite.jobs(a.splits)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(record,work))
    save(a.out,rows)
    print('accepted',sum(bool(r.get('accepted')) for r in rows),'of',len(rows))
