"""Audit the source partitions, then supply their dominant native plane rulers for a bounded replay."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_wall_rulers import reference_chart
from benchmark.floating_mixed_wall_refit import OUT, REVIEW, BASE, PREFIX, jobs, data, read, save, project, combined, summarize
from benchmark.floating_one_face_control import extend


def native(group,case):
    name=case['id'];rel=Path('sky')/name;frame=Path(case['frame'])
    src=OUT/group/'source'/name
    if not (src/'region-ids.png').is_file():
        return None
    regions=np.asarray(Image.open(src/'region-ids.png'))
    packet=read(OUT/group/'split/charts'/rel/'charts.json')
    source=read(OUT/group/'visible-patches/charts'/rel/'charts.json')
    by={c['id']:c for c in source['charts']}
    labels=read(frame/'labels.json')
    faces={f['id']:f for f in labels['faces']}
    blocks={b['id']:b for b in labels['blocks']}
    kinds={k:b.get('geometry_kind','full_cube') for k,b in blocks.items()}
    pose=np.array(labels['camera']['world_to_pixel_homogeneous'])
    def key(f):
        normal=np.asarray(f['normal_world']);depth=np.array(f['corners_world'])@normal
        return f['direction']+':'+str(round(float(np.median(depth)),3))
    face_groups={k:key(f) for k,f in faces.items()}
    with np.load(frame/'labels.npz') as f:
        ids=f['face_id']
    assert ids.shape==regions.shape
    def votes(mask):
        fids,counts=np.unique(ids[mask],return_counts=True)
        planes=Counter()
        for i,n in zip(fids,counts):
            planes[face_groups.get(i,'background')]+=int(n)
        return planes
    before=votes(regions>0); dominant=sum(votes(regions==k).most_common(1)[0][1] for k in np.unique(regions) if k)
    report=dict(mask_pixels=int((regions>0).sum()),original_dominant=before.most_common(4),part_dominant_pixels=dominant)
    rgb=np.asarray(Image.open(case['variants']['sky']['image']).convert('RGB'))
    updated=[];pieces=[]
    for p in packet['charts']:
        if '~r' not in p['id']:
            updated.append(p);continue
        k=int(p['id'].split('~r')[-1]);mask=regions==k;cnt=votes(mask)
        plane=next((k for k,_ in cnt.most_common() if k.split(':')[0] in ('north','south','west','east')),None)
        chosen=[f for fid,f in faces.items() if face_groups[fid]==plane and kinds[f['block']]=='full_cube']
        native_grid=reference_chart(chosen,kinds,pose) if chosen else None
        row=dict(id=p['id'],dominant=cnt.most_common(3),reference_plane=plane)
        if native_grid:
            h,error=native_grid
            new=extend(p,h,mask,rgb)
            row.update(reference_cells=len(new['cells']),checked_corner_error_px=error)
            updated.append(new)
            if by[p['id']]['H']!=p['H']:
                probe=by[p['id']]
                center=project(np.array(probe['H']),np.asarray([[c['u']+.5,c['v']+.5] for c in probe['cells']])).astype(int)
                natives=ids[center[:,1],center[:,0]]
                witness=[f for f in natives if face_groups.get(f)==plane]
                grid=project(np.linalg.inv(probe['H']),np.array([faces[f]['corners_px'] for f in set(witness)]))
                xx,yy=grid.transpose(2,0,1)
                areas=abs(np.sum(xx*np.roll(yy,-1,axis=1)-yy*np.roll(xx,-1,axis=1),axis=1))*.5
                row['source_correction']=dict(cells=len(center),principal_centers=len(witness),
                    principal_unique=len(set(witness)),median_cells_per_native_face=float(np.median(areas)),
                    native_center_materials=Counter(blocks[faces[f]['block']]['block_id'] for f in natives if f))
        else:
            updated.append(p)
        pieces.append(row)
    root=OUT/group/'native-rulers'
    save(root/'charts'/rel/'charts.json',dict(packet,charts=updated))
    with patch.object(combined,'inputs',data):
        combined.run((group,case,root/'charts',root/'assembly'))
    scored=summarize([case],dict(base=BASE/group/'large-contraction/assembly',native=root/'assembly'),PREFIX)['rows'][0]
    print(name,report,scored['score']['native']['geometry']['strict'],scored['paired']['native'],flush=True)
    return dict(scored,panels=pieces,ownership=report,group=group)


if __name__=='__main__':
    cv2.setNumThreads(1)
    rows=[row for g,c in jobs() if (row:=native(g,c))]
    save(REVIEW/'native.json.gz',rows)
