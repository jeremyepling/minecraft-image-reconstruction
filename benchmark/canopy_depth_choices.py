"""Generate wide plane choices on train and the earlier native transfer development set."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark.texture_camera_eval import DATA, ROOT, STABLE, checked, read, save
from benchmark.visible_depth_transfer import OUT, rgb_path
from benchmark.camera_metrics import registration
from engine.atlas_plane_depth import _atlas, options
from engine.plane_repeat_evidence import near_levels, repeat_scores
from engine.visible_placement import _estimate_plane_cells
from engine.visible_raster import SAMPLE_STRIDE, observed_support
from training.visible_labels import visible_labels
from PIL import Image
import numpy as np


DEST=ROOT/'runs/plane-choices'


def extract(job):
    split, f = job
    key = f['id']
    dest = DEST/split/(key+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if split == 'transfer':
        png = checked(rgb_path(f),f['files']['rgb.png'])
        parts = read(OUT/'upstream'/key/'surfaces.json')['planes']
        P = f['camera']['P']
    else:
        folder=DATA/'data'/split/'frames'/key
        png=checked(folder/'rgb.png',f['rgb_sha256'])
        parts=f['parts']
        P=f['camera']['P']
    with Image.open(png) as opened: image=opened.convert('RGB')
    rgb=np.asarray(image)
    rows=[]
    if P is not None and parts:
        atlas=_atlas('26.3')
        if atlas is None: raise ValueError('Missing atlas')
        P=np.asarray(P)/np.linalg.norm(np.asarray(P)[2,:3])
        eye=np.linalg.solve(P[:,:3],-P[:,3])
        shape=tuple((i+1)//SAMPLE_STRIDE for i in rgb.shape[:2])
        masks=[observed_support(p['support'],shape,image.size) if p.get('support') else np.zeros(shape,bool) for p in parts]
        def collect(part,mask,P,eye,rays,axis,baseline,scene):
            opts=options(mask,P,eye,rays,axis,baseline,rgb,atlas,scene,radius=18,
                         extra_levels=near_levels(eye[axis],baseline))
            scene_level=round(float(eye[axis]+np.median(rays[mask,axis])*scene))
            levels=[r['level'] for r in opts]
            for opt,(hi,lo) in zip(opts,repeat_scores(rgb,mask,P,eye,axis,levels),strict=True):
                opt.update(repeat_high=hi,repeat_low=lo)
            rows.append(dict(id=part['id'],axis=axis,baseline=baseline,scene=scene_level,eye=float(eye[axis]),
                             n=int(mask.sum()),choices=opts))
            return baseline
        _estimate_plane_cells(image,parts,masks,P,eye,collect)
        # Oracle labels are opened only after all image-derived proposals are preserved.
        if split=='transfer':
            pose=read(checked(png.with_suffix('.json'),f['files']['camera.json']))
            raw=read(checked(png.with_name(f['source_id']+'-blocks.json'),f['files']['world-readback.json']))
            states={tuple(b[k] for k in 'xyz'):b for b in raw}
            labels,ids,_=visible_labels(pose,states,(0,0,0))
        else:
            pose=read(folder/'camera.json')
            labels=read(folder/'labels.json')
            with np.load(folder/'labels.npz') as a:ids=a['face_id']
        ids=ids[2::4,2::4]
        T=registration(P,pose)
        lookup={}
        for face in labels['faces']:
            axis=int(np.argmax(abs(np.asarray(face['normal_world'])@T[:3,:3])))
            p=(np.asarray(face['corners_world'][0])-T[:3,3])@T[:3,:3]
            direction=face['direction']
            lookup[face['id']]=(axis,int(round(p[axis])),direction)
        by_id={p['id']:m for p,m in zip(parts,masks)}
        for row in rows:
            mask=by_id[row['id']]
            values,counts=np.unique(ids[mask],return_counts=True)
            truth={}
            for v,n in zip(values,counts):
                if v and lookup[v][0]==row['axis']:
                    level=lookup[v][1]
                    truth[level]=truth.get(level,0)+int(n)
            row['truth']=truth
    save(dest,dict(id=key,split=split,family=f.get('family',f.get('shape')),rows=rows))
    return key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=('train','transfer'),required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    frames=read(OUT/'screen.json')['selected'] if args.split=='transfer' else [
        read(path) for path in sorted((STABLE/'runs/visible-depth-rounds/inputs').glob('train-*.json'))]
    with ProcessPoolExecutor(args.workers) as pool:
        for key in pool.map(extract,[(args.split,f) for f in frames]):print(key,flush=True)


if __name__=='__main__':main()
