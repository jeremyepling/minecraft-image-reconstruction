"""Replay per-surface observed cell footprints before and after grouping. Native IDs grade only."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import combinations
from hashlib import sha256
import json
import argparse
import numpy as np
from PIL import Image
from benchmark import full_block_lattice_run as run
from benchmark.expanded_baseline import SOURCE,frames,read,save
from benchmark.full_block_lattice import line_camera,mask_parts,occupancy,atlas
from benchmark.mask_blocks import decoded_pack
from benchmark.typed_structures import typed_pack
from engine.placed_materials import assign


def grade(observations,groups,raster,labels):
    owner_at=np.array([0]+[f['block'] for f in labels['faces']])
    partial={b['id'] for b in labels['blocks'] if b.get('geometry_kind','full_cube')!='full_cube'}
    votes=[]
    for points in observations:
        if not len(points):
            votes.append((0,0)); continue
        native_ids=raster.ravel()[points]
        owners=owner_at[native_ids]
        owner,n=Counter(owners.tolist()).most_common(1)[0]
        if owner and n/len(owners)>=.65:
            face=Counter(native_ids[owners==owner].tolist()).most_common(1)[0][0]
            votes.append((owner,face))
        else:
            votes.append((0,0))
    count=Counter(observations=len(votes),blank_or_mixed=sum(not q[0] for q in votes),groups=len(groups))
    represented=defaultdict(set)
    valid_merges=set()
    for index,group in enumerate(groups):
        valid=[votes[i] for i in group if votes[i][0]]
        targets={own for own,face in valid}
        if len(targets)>1:
            count['neighbor_over_merges']+=1
        for own in targets:
            represented[own].add(index)
        for (oa,fa),(ob,fb) in combinations(valid,2):
            if oa!=ob:
                count['different_block_pairs_joined']+=1
            elif fa!=fb:
                count['same_block_distinct_face_pairs_joined']+=1
                if oa in partial and len(targets)==1:
                    valid_merges.add(oa)
        if len(targets)==1 and len({f for _,f in valid})>=2:
            count['pure_multiface_groups']+=1
    count.update(represented=len(represented),partial_represented=len(represented.keys()&partial),
                 oversplit=sum(len(v)-1 for v in represented.values()),
                 partial_oversplit=sum(len(v)-1 for p,v in represented.items() if p in partial),
                 partial_true_multiface_merges=len(valid_merges),partial_targets=len(partial))
    return dict(count)


def one(entry):
    out=run.RUN/entry['dataset']/entry['id']
    if (out/'grouping.json').exists():
        return
    dataset,key=entry['dataset'],entry['id']
    is_decor=dataset=='decorative-builds-v1'
    evidence=read(out/'evidence.json.gz')
    row=read(out/'result.json')
    if not is_decor and not row['rescued']:
        return
    root=run.BASE/dataset/key
    hand=read(root/'handoffs/placement-input.json.gz')
    P=(hand['camera'] or {}).get('P')
    anchors=evidence.get('anchors',[])
    rgb=Path(entry['frame'])/'rgb.png'
    image=Image.open(rgb).convert('RGB')
    mapping=defaultdict(set)
    retained=defaultdict(set)
    cells={}
    if anchors:
        eye=np.array(evidence['eye'])
        if P is None:
            lines=read(root/'handoffs/mlsd.json.gz')['segments']
            linear=line_camera(lines,image.size)
            P=np.column_stack((linear,-linear@eye))
        P=np.array(P)
        P/=np.linalg.norm(P[2,:3])
        parts=mask_parts(hand['parts'],image.size,P[:,:3])
        def keep(surface,cell,axis,level,y,x):
            k=surface,cell
            mapping[k].update(((y*4+2)*image.width+x*4+2).tolist())
            retained[k].add((axis,round(level,4)))
        cells,_,_=occupancy(parts,anchors,eye,P[:,:3],observe=keep)
        original=set(decoded_pack(root/'reconstruction.zip'))
        assert original|cells.keys()==decoded_pack(out/'proposal.zip').keys()
    # Save each actual observed source footprint (stride 4), with original mask IDs untouched.
    fragments=list(mapping)
    observations=[np.array(sorted(mapping[k]),dtype=int) for k in fragments]
    by_cell=defaultdict(list)
    for i,(surface,cell) in enumerate(fragments):
        by_cell[cell].append(i)
    groups=list(by_cell.values())
    save(out/'grouped-observations.json.gz',dict(source_sha256=entry['rgb_sha256'],stride=4,width=image.width,
        fragments=[dict(surface=s,local=c,planes=sorted(retained[s,c]),native_pixel_indices=ps.tolist())
                   for (s,c),ps in zip(fragments,observations)],groups=groups))
    # Ordinary material matcher consumes the recovered occupied group coordinates and the image.
    material=None
    if row['rescued']:
        assigned,methods=assign(image,dict(P=P.tolist()),cells,atlas())
        pack,_=typed_pack(assigned)
        (out/'ordinary-material-on-grouped.zip').write_bytes(pack)
        material=(pack,methods)
    # Reference boundary. Native visible IDs establish group purity and recovery.
    labels=read(Path(entry['frame'])/'labels.json')
    with np.load(Path(entry['frame'])/'labels.npz') as a:
        raster=a['face_id']
    counts=read(root/'counts.json')
    faces=sorted([f for s in counts['surfaces'] for prop in ('visible_cells','under_five_px_faces') for f in s.get(prop,[])],
                 key=lambda f:f['face_index'])
    existing=counts.get('block_groups',[])
    from engine.geometry import project,transform
    old=[]
    for i,f in enumerate(faces):
        assert i==f['face_index']
        t=(np.arange(7)+.5)/7
        xy=project(transform(np.array(f['quad'])),np.stack(np.meshgrid(t,t),axis=-1))
        x,y=np.moveaxis(np.floor(xy).astype(int),-1,0)
        old.append((y.clip(0,899)*image.width+x.clip(0,1439)).ravel())
    data=dict(id=key,dataset=dataset,split=entry['split'],source_gate=row['changed'],
              ordinary=grade(old,[g['face_indices'] for g in existing],raster,labels),
              separate=grade(observations,[[i] for i in range(len(fragments))],raster,labels),
              grouped=grade(observations,groups,raster,labels))
    if material is not None:
        payload,methods=material
        scored,_=run.scorer.evaluate_pack(entry,payload,dict(camera_status='supported',P=P.tolist()),assignment=methods)
        assert scored['geometry']==row['source']['geometry']
        data['material']=run.compact_row(scored)
    save(out/'grouping.json',data)


def summary():
    rows=[read(f) for f in run.RUN.glob('*/*/grouping.json')]
    result={}
    for dataset in run.COHORTS:
        ds=[r for r in rows if r['dataset']==dataset]
        cohorts=dict(all=ds,**{split:[r for r in ds if r['split']==split] for split in {r['split'] for r in ds}})
        out={}
        for split,group in cohorts.items():
            out[split]={'tested':len(group)}
            for arm in ('ordinary','separate','grouped'):
                total=Counter()
                for r in group:
                    total.update(r[arm])
                out[split][arm]=dict(total)
        result[dataset]=out
    material_by={r['id']:r['material'] for r in rows if 'material' in r}
    paired=read(run.REVIEW/'cases.json.gz')
    for dataset in run.COHORTS:
        all_cases=[r for r in paired if r['dataset']==dataset]
        result[dataset]['fallback_with_ordinary_material']={k:run.typed.pooled_scores(
            [material_by.get(r['id'],r['baseline'])[k] for r in all_cases]) for k in ('geometry','scores')}
    save(run.REVIEW/'grouping.json',result)
    save(run.REVIEW/'grouping-cases.json.gz',rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        for _ in pool.map(one,list(frames(SOURCE,run.COHORTS))):
            pass
    summary()


if __name__=='__main__':
    main()
