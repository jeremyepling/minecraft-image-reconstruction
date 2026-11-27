"""Identify every decorative multi-face partial join and check its exported coordinate."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from benchmark import view_atlas_lattice_run as run
from benchmark.expanded_baseline import SOURCE,frames,read,save
from benchmark.typed_structures import aligned_sites
from benchmark.camera_metrics import registration
from benchmark.full_block_lattice import line_camera


def main():
    rows=[]
    for e in frames(SOURCE,['decorative-builds-v1']):
        ds,key=e['dataset'],e['id']
        frame=Path(e['frame'])
        labels=read(frame/'labels.json')
        pose=read(frame/'camera.json')
        with np.load(frame/'labels.npz') as raster:front=raster['face_id']
        owners=np.array([0]+[f['block'] for f in labels['faces']])
        for arm,root in [('grid',run.OLD),('point',run.RUN.parent/'view-atlas-lattice-point-01'),('finite',run.RUN)]:
            folder=root/ds/key
            d=read(folder/'grouped-observations.json.gz')
            row=read(folder/'result.json')
            for group in d['groups']:
                votes=[]
                for i in group:
                    f=d['fragments'][i]
                    ids=front.ravel()[f['native_pixel_indices']]
                    if not len(ids):continue
                    owner,count=Counter(owners[ids].tolist()).most_common(1)[0]
                    if not owner or count<len(ids)*.65:continue
                    face,n=Counter(ids[owners[ids]==owner].tolist()).most_common(1)[0]
                    votes.append((owner,int(face),n,f))
                if len({v[0] for v in votes})!=1 or len({v[1] for v in votes})<2:continue
                owner=votes[0][0]
                b=labels['blocks'][owner-1]
                if b.get('geometry_kind','full_cube')=='full_cube':continue
                P=read(run.release.BASE/ds/key/'build-manifest.json')['placement']['P']
                if P is None:
                    linear=line_camera(read(run.release.BASE/ds/key/'handoffs/mlsd.json.gz')['segments'],(1440,900))
                    P=np.column_stack((linear,-linear@np.array(read(folder/'evidence.json.gz')['eye'])))
                T=registration(np.asarray(P),pose)
                cell=votes[0][3]['local']
                reg=aligned_sites([cell],T,(0,0,0))[0]
                rel=np.array(reg)+row['proposal']['translation']
                rows.append(dict(id=key,arm=arm,truth=b,local=cell,registered=list(reg),relative=rel.tolist(),
                    correct_registered=list(reg)==b['xyz_world'],correct_relative=rel.tolist()==b['xyz_world'],
                    observations=[dict(surface=f['surface'],face=face,frontmost_majority_pixels=n,total=len(f['native_pixel_indices']))
                                  for _,face,n,f in votes]))
    save(run.REVIEW/'partial-joins.json',rows)
    print([(r['arm'],r['truth']['xyz_world'],r['truth']['geometry_kind'],r['correct_registered'],r['correct_relative']) for r in rows])


if __name__=='__main__':
    main()
