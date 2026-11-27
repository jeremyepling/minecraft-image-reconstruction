"""Audit the two grid predictions against native visible owner pixels after export."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from benchmark import view_atlas_lattice as experiment,view_atlas_lattice_run as stage
from benchmark import full_block_lattice_groups as groups,full_block_lattice_run as release
from benchmark.expanded_baseline import read,save
from benchmark.camera_metrics import registration
from benchmark.typed_structures import aligned_sites
from engine.geometry import project,transform


def observed(entry):
    ds,key=entry['dataset'],entry['id']
    out=stage.RUN/ds/key
    if not (out/'result.json').exists():
        return
    if ds=='decorative-builds-v1':
        experiment._LAST['rgb']=np.asarray(Image.open(Path(entry['frame'])/'rgb.png').convert('RGB'))
        with patch.object(groups.run,'RUN',stage.RUN),patch.object(groups,'occupancy',experiment.group):
            groups.one(entry)
    if ds==release.FIXED:
        labels,raster,T=release.fixed.fixed_reference(entry)
        src_camera=release.fixed._jobs()[key][3]['status']
        pose=None
    else:
        root=release.BASE/ds/key
        src_camera=read(root/'build-manifest.json')['placement']
        labels=read(Path(entry['frame'])/'labels.json')
        with np.load(Path(entry['frame'])/'labels.npz') as native:raster=native['face_id']
        pose=read(Path(entry['frame'])/'camera.json')
    result=dict(dataset=ds,id=key,split=entry['split'],floor_rejected=src_camera['P'] is None)
    for arm,work in [('grid',stage.OLD/ds/key),('joint',out)]:
        evidence=read(work/'evidence.json.gz')
        row=read(work/'result.json')
        anchors=evidence.get('anchors',[])
        rows=[]
        if anchors:
            eye=np.array(evidence['eye'])
            if ds!=release.FIXED:
                P=src_camera['P']
                if P is None:
                    linear=experiment.grid.line_camera(read(root/'handoffs/mlsd.json.gz')['segments'],(1440,900))
                    P=np.column_stack((linear,-linear@eye))
                T=registration(np.asarray(P),pose)
        for a in anchors:
            q=np.array(a['quad'])
            t=(np.arange(9)+.5)/9
            xy=project(transform(q),np.stack(np.meshgrid(t,t),axis=-1)).astype(int)
            x,y=np.moveaxis(xy,-1,0)
            native,n=Counter(raster[y.clip(0,899),x.clip(0,1439)].ravel().tolist()).most_common(1)[0]
            item=dict(corr=a['corr'],rescued=row['rescued'],accepted=row['changed'],full_unit=False,
                      correct_signed_direction=False,intended_registered=False,intended_relative=False)
            if native:
                f=labels['faces'][native-1]; block=labels['blocks'][f['block']-1]
                true=np.array(f['corners_px'])
                width=float(np.linalg.norm(true-np.roll(true,1,axis=0),axis=1).mean())
                corner_error=float(min(np.linalg.norm(q-np.roll(order,i,axis=0),axis=1).mean()
                                       for order in (true,true[::-1]) for i in range(4))/max(1,width))
                ratio=float(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1).mean()/max(1,width))
                ax=a['axis']
                ref_pt=np.array(block['xyz_world'])
                xyz=np.array(a['xyz'])
                center=xyz.mean(0)+eye
                sign=np.sign(-xyz[0,ax])
                center[ax]-=sign*.02
                cell=tuple(np.floor(center+1e-7).astype(int).tolist())
                registered=np.array(aligned_sites([cell],T,(0,0,0))[0])
                relative=registered+row['proposal']['translation']
                full=block.get('geometry_kind','full_cube')=='full_cube'
                item.update(majority_full_face=full and n>=65,full_unit=full and n>=65 and corner_error<=.20,
                    correct_signed_direction=bool(np.array_equal(T[:3,ax]*sign,f['normal_world'])),
                    intended_registered=bool(np.array_equal(registered,ref_pt)),intended_relative=bool(np.array_equal(relative,ref_pt)),
                    intended_shift=(ref_pt-registered).tolist(),error=round(corner_error,5),ratio=round(ratio,5),
                    adjusted=a.get('adjusted',False),correct_id='minecraft:'+a['block']==block['block_id'])
            rows.append(item)
        result[arm]=rows
        if ds=='decorative-builds-v1':
            result[arm+'_group']=read(work/'grouping.json')['grouped']
    if ds=='decorative-builds-v1' and key=='test-walled-garden-1-warm-high-right':
        example=read(release.ROOT/'benchmark/reviews/full-block-lattice-01/partial-example.json')
        target=example['truth']['id']
        owner_lookup=np.array([0]+[f['block'] for f in labels['faces']])
        for arm,work in [('grid',stage.OLD/ds/key),('joint',out)]:
            sel=[]
            for f in read(work/'grouped-observations.json.gz')['fragments']:
                front=raster.ravel()[f['native_pixel_indices']]
                if len(front) and np.mean(owner_lookup[front]==target)>=.65:
                    sel.append(dict(surface=f['surface'],cell=f['local'],plane=f['planes'],hits=len(front)))
            result[arm+'_slab']=sel
    save(out/'anchor-comparison.json.gz',result)
    return key


def aggregate():
    rows=[read(p) for p in stage.RUN.glob('*/*/anchor-comparison.json.gz')]
    out={}
    for ds in (release.FIXED,*release.COHORTS):
        selected=[r for r in rows if r['dataset']==ds]
        by={}
        for category,case in (('all',selected),('floor_rejected',[r for r in selected if r['floor_rejected']])):
            group={}
            for arm in ('grid','joint'):
                anchors=[a for r in case for a in r[arm]]
                valid=[a for a in anchors if a.get('majority_full_face')]
                group[arm]=dict(anchors=len(anchors),full_unit=sum(a['full_unit'] for a in anchors),dominant_full=len(valid),
                    correct_signed_axis=sum(a['correct_signed_direction'] for a in valid),
                    own_registered=sum(a['intended_registered'] for a in valid),own_relative=sum(a['intended_relative'] for a in valid),
                    exact_own_id=sum(a['correct_id'] for a in valid),
                    median_corner_error=float(np.median([a['error'] for a in valid])) if valid else None,
                    median_scale_ratio=float(np.median([a['ratio'] for a in valid])) if valid else None,
                    best_single_shift=sum(max(Counter(tuple(a['intended_shift']) for a in r[arm] if a.get('majority_full_face')).values(),default=0) for r in case))
            by[category]=group
        if ds=='decorative-builds-v1':
            for split,rs in [('all',selected),('test',[r for r in selected if r['split']=='test'])]:
                groups_out={}
                for arm in ('grid','joint'):
                    counts=Counter()
                    for r in rs:counts.update(r[arm+'_group'])
                    groups_out[arm]=dict(counts)
                by['grouping_'+split]=groups_out
            by['slab']={k:v for r in selected if 'joint_slab' in r for k,v in r.items() if k in ('grid_slab','joint_slab')}
        out[ds]=by
    save(stage.REVIEW/'anchors-and-grouping.json',out)
    save(stage.REVIEW/'anchor-cases.json.gz',rows)


def main():
    with ProcessPoolExecutor(2) as pool:
        for _ in pool.map(observed,[e for e in stage.all_frames() if not (stage.RUN/e['dataset']/e['id']/'anchor-comparison.json.gz').exists()]):pass
    aggregate()


if __name__=='__main__':
    main()
