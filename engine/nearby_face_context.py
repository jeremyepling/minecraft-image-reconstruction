"""Compare local depth alternatives with equivalent shifts on adjacent observed cells."""
import numpy as np


def enrich(data):
    edges,x=data['edges'],data['features']
    if not len(edges):
        return data
    points=data['points']
    s,t,a=edges.T
    origin=points[s]
    displacement=points[t]-origin
    level=displacement[np.arange(len(a)),a]
    # Coplanar neighbours at one or two tile steps should corroborate a genuine
    # surface-level change. Compare the full grid shift separately, since an
    # oblique camera can also move the tangential index across a boundary.
    metrics=np.column_stack((x[:,2],x[:,4],x[:,16],x[:,18],x[:,22],x[:,24],
                             x[:,58],x[:,59],x[:,60],x[:,61]))
    exact={}
    near={}
    for i,(xyz,d,axis,offset) in enumerate(zip(origin,displacement,a,level)):
        exact[(*xyz,*d,axis)]=i
        key=(*xyz,axis,offset)
        k=near.get(key)
        if k is None or x[i,2]>x[k,2]:
            near[key]=i
    n,dim=metrics.shape
    columns=[]
    for table,keys in ((exact,np.column_stack((displacement,a))),(near,np.column_stack((a,level)))):
        counts=np.zeros(n)
        total=np.zeros((n,dim))
        strongest=np.zeros((n,dim))
        for direction in range(3):
            candidates=np.flatnonzero(a!=direction)
            for step in (-2,-1,1,2):
                xyz=origin[candidates].copy()
                xyz[:,direction]+=step
                request=np.column_stack((xyz,keys[candidates]))
                matches=np.array([table.get(tuple(row),-1) for row in request],dtype=int)
                good=matches>=0
                ids=candidates[good]
                neighbors=metrics[matches[good]]
                counts[ids]+=1
                total[ids]+=neighbors
                strongest[ids]=np.maximum(strongest[ids],neighbors)
        columns.extend((counts[:,None],total/np.maximum(1,counts[:,None]),strongest))
    return dict(data,features=np.column_stack((x,*columns)).astype(np.float32))

