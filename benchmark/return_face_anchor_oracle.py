"""Diagnostic only: supply the true dominant plane of each proposed perpendicular face."""
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import numpy as np
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.harbor_carrier_audit import native_face_planes
from benchmark.return_face_round import CONTROL, RUN, PC, source, read, save
from benchmark import return_face_proposals as model


def run(row):
    image,parts,camera,planarity=source(row)
    key=row['id']
    if row['dataset']=='fixed-104':
        e=next(e for e in fixed.fixed_frames() if e['id']==key)
        labels,ids,T=fixed.fixed_reference(e)
    else:
        labels=read(PC/row['reference']['labels.json']['path'])
        with np.load(PC/row['reference']['labels.npz']['path']) as archive:
            ids=archive['face_id']
        T=row['baseline']['registration']['transform']
    planes=native_face_planes(labels,np.asarray(T))
    native=np.array([[-1,0]]+[planes[i] for i in range(1,len(planes)+1)])[ids]
    def ideal(mask,axis,*_):
        # Native pixels provide the return's dominant measured axis and level.
        dense=np.repeat(np.repeat(mask,4,axis=0),4,axis=1)
        pixels=native[:dense.shape[0],:dense.shape[1]][dense]
        levels,n=np.unique(pixels[pixels[:,0]==axis,1],return_counts=True)
        if len(levels) and max(n)>.5*len(pixels):
            return [dict(level=int(levels[n.argmax()]),method='visible-reference-majority',strength=1.)]
        return []
    with patch.object(model,'anchors',ideal):
        result,_=model.propose(image,parts,np.array(camera['P']),planarity)
    save(RUN/key/'ideal-anchors.json',result)
    print(key,len(result['options']),flush=True)

if __name__=='__main__':
    with ProcessPoolExecutor(5) as pool:
        list(pool.map(run,read(CONTROL)['pilot']['frames']))
