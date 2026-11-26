"""Measure native top-face observability and texture evidence on earlier development canopies."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
import numpy as np
from benchmark.canopy_composition import with_edges
from benchmark.camera_metrics import registration
from benchmark.texture_camera_eval import ROOT,read,save
from benchmark.visible_depth_transfer import rgb_path,OUT
from engine.camera_atlas_floor import _atlas
from engine.grazing_faces import proposals
from engine.visible_edge_faces import native_support
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE
from training.visible_labels import visible_labels


def one(f):
    key=f['id'];png=rgb_path(f)
    with Image.open(png) as photo:image=photo.convert('RGB')
    parts=read(OUT/'upstream'/key/'surfaces.json')['planes']
    counts=read(OUT/'upstream'/key/'counts.json')
    camera=f['camera'];P=np.asarray(camera['P'])
    cells,status=with_edges(image,'26.3',parts,counts,camera)
    opts=[proposals(image,parts,P,cells,status['plane_offsets'],_atlas('26.3'),union=union) for union in (False,True)]
    pose=read(png.with_suffix('.json'))
    raw=read(png.with_name(f['source_id']+'-blocks.json'))
    states={tuple(b[k] for k in 'xyz'):b for b in raw}
    labels,face_ids,_=visible_labels(pose,states,(0,0,0))
    top=max(b['xyz_world'][1] for b in labels['blocks'])
    targets={tuple(b['xyz_world']) for b in labels['blocks']}
    faces={tuple(b['xyz_world']):b for b in labels['faces'] if b['xyz_world'][1]==top and b['direction']=='up'}
    T=registration(P,pose)
    def world(p):
        return tuple(np.floor(T[:3,:3]@(np.array(p)+.5)+T[:3,3]).astype(int).tolist())
    correct={world(p) for p in cells}
    missing=faces.keys()-correct
    all_masks=np.logical_or.reduce([native_support(p['support'],image.size) for p in parts if p.get('support')])
    native=P.copy();native[:2]*=SAMPLE_STRIDE
    owner=_frontmost_cells(sorted(cells),native,np.linalg.solve(P[:,:3],-P[:,3]),face_ids.shape)
    in_mask=np.bincount(face_ids[all_masks],minlength=len(labels['faces'])+1)
    free=np.bincount(face_ids[all_masks&(owner<0)],minlength=len(labels['faces'])+1)
    metrics=dict(visible=len(faces),already_correct=len(set(faces)&correct),missing=len(missing),
                 missing_face_pixels=[faces[p]['visible_pixels'] for p in missing],
                 mask_witnesses_14=sum(int(in_mask[faces[p]['id']]>=14) for p in missing),
                 unclaimed_witnesses_14=sum(int(free[faces[p]['id']]>=14) for p in missing))
    for method,rows in zip(('mask','union'),opts):
        candidates={world(p):r for p,r in rows.items()}
        metrics[method]=dict(candidate_count=len(rows),missing_top_proposed=len(candidates.keys()&missing),
            strong_top=sum(p in missing and r['score']>.74 and r['phase']>.16 for p,r in candidates.items()),
            strong_wrong=sum(p not in targets and r['score']>.74 and r['phase']>.16 for p,r in candidates.items()))
    return dict(id=key,metrics=metrics)


if __name__=='__main__':
    frames=[f for f in read(OUT/'screen.json')['selected'] if 'canopy-' in f['id']]
    with ProcessPoolExecutor(3) as pool:rows=list(pool.map(one,frames))
    dest=ROOT/'benchmark/reviews/grazing-roof-01'
    dest.mkdir(parents=True,exist_ok=True);save(dest/'oracle.json',rows)
    for k in ('mask','union'):
        c=Counter()
        for r in rows:c.update(r['metrics'][k])
        print(k,c)
    pixels=np.concatenate([r['metrics']['missing_face_pixels'] for r in rows])
    print('missing top face pixels',len(pixels),'median',np.median(pixels),'at most 14',int((pixels<14).sum()))
