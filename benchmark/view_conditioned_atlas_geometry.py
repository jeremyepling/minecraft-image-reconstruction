"""Test photo-atlas grid scale/origin and plane ownership on visible source samples.

Only the final grid quantizer and boundary pixel choice vary on frozen ordinary
camera, material and frontmost plane proposals. Native face IDs enter grading.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import numpy as np
from PIL import Image
from benchmark.view_conditioned_atlas_run import BASE,OUT,REVIEW,entries,inputs,read,save,decoded_pack,_export_atlas
from benchmark.expanded_scores import evaluate_pack,load_reference
from benchmark.typed_structures import typed_pack,pooled_scores
from engine.placed_materials import Catalog
from engine.projected_appearance import Scene
from engine.thin_atlas_faces import _corr

OFFSETS=np.array([[x,y] for x in (-1/3,0,1/3) for y in (-1/3,0,1/3)])

def coordinates(scene,xy,cell,axis):
    ray=xy@scene.inv[:,:2].T + scene.inv[:,2]
    level=cell[axis]+int(scene.eye[axis]>cell[axis]+1)
    free=[d for d in range(3) if d!=axis]
    return scene.eye[free]+(level-scene.eye[axis])*ray[...,free]/ray[...,axis,None]

def sample_uv(uv):
    q=np.floor(uv*16).astype(int)%16
    return q[...,0]+16*q[...,1]

def fit(scene,idx,cell,axis,tiles,offsets):
    y,x=np.divmod(idx,scene.shape[1])
    xy=np.column_stack((x,y))*scene.step+.5
    uv=coordinates(scene,xy[:,None]+offsets,cell,axis)
    rgb=scene.rgb[y,x].mean(1)
    features=tiles[:,sample_uv(uv)].mean(-1)
    rot=int(_corr(features,rgb).argmax())
    tmpl=features[rot]
    gain=np.clip(((tmpl-tmpl.mean())@(rgb-rgb.mean()))/max(1.,((tmpl-tmpl.mean())**2).sum()),.25,1.12)
    return uv,rgb,tiles[rot],gain,rgb.mean()-gain*tmpl.mean()

def plane_groups(scene,ids):
    groups=defaultdict(list)
    for n,idx in scene.owned.items():
        cell=scene.cells[n]
        for axis in np.unique(scene.axes[idx]):
            part=idx[scene.axes[idx]==axis]
            groups[(int(axis),cell[axis],ids[cell])].append((n,part))
    return groups

def source(scene,ids,atlas,offsets):
    cat=Catalog(atlas)
    proposals={}
    records=[]
    planes=np.full(scene.shape,-1,int)
    models=[]
    targets=dict(ids)
    evidence=defaultdict(dict)
    for (axis,level,block),faces in plane_groups(scene,ids).items():
        members=np.unique([n for n,_ in faces])
        idx=np.concatenate([part for _,part in faces])
        if len(idx)<80 or block not in cat.blocks:
            continue
        cell=scene.cells[members[0]]
        side=('top' if scene.eye[1]>level+1 else 'bottom') if axis==1 else 'wall'
        sprite=np.unique(cat.sprites[cat.valid[side] & (cat.candidates==cat.blocks.index(block))])
        if not len(sprite):
            continue
        planes.ravel()[idx]=len(models)
        chosen=np.sort(idx)[np.linspace(0,len(idx)-1,min(384,len(idx)),dtype=int)]
        maps=cat.maps[sprite,0].reshape(-1,256)
        uv,rgb,tile,gain,bias=fit(scene,chosen,cell,axis,maps,offsets)
        models.append((cell,axis,tile,gain,bias))
        if len(members)<4 or rgb.std()<2:
            continue
        free=[d for d in range(3) if d!=axis]
        coordinates0=np.array([scene.cells[n] for n in members])[:,free]
        start=coordinates0.min(0)
        extent=np.ptp(coordinates0,axis=0)+1
        relative=uv-start
        options=[((1.,1.),(0.,0.))]
        for d,size in enumerate(extent):
            if size<3:
                continue
            for delta in (-1,1):
                scale=np.ones(2);scale[d]+=delta/size
                options.append((tuple(scale),(0.,0.)))
            for delta in (-.25,.25):
                shift=np.zeros(2);shift[d]=delta
                options.append(((1.,1.),tuple(shift)))
        if len(options)==1:
            continue
        # The sprite is an ordinary predicted ID. Refit orientation for each grid.
        scores=[]
        for scale,shift in options:
            pr=maps[:,sample_uv(relative*scale+shift)].mean(-1)
            orientation=_corr(pr[:,::2],rgb[::2]).argmax()
            scores.append(min(float(_corr(pr[orientation,::2],rgb[::2])),
                              float(_corr(pr[orientation,1::2],rgb[1::2]))))
        winner=int(np.argmax(scores))
        if scores[winner]<.38 or scores[winner]-scores[0]<.12:
            winner=0
        scale,shift=options[winner]
        records.append(dict(axis=axis,plane=int(level),material=block,extent=extent.tolist(),
                            proposed_extent=np.rint(extent*scale).astype(int).tolist(),
                            shift=list(shift),base=scores[0],selected=scores[winner],changed=bool(winner),
                            plane_number=len(models)-1,start=start.tolist(),scale=list(scale)))
        if not winner:
            continue
        for n,part in faces:
            y,x=np.divmod(part,scene.shape[1])
            q=coordinates(scene,np.column_stack((x,y))*scene.step+.5,cell,axis)
            alternatives=np.floor((q-start)*scale+shift).astype(int)+start
            # Only export cells supported by an observed source pixel on this face.
            for pos,count in zip(*np.unique(alternatives,axis=0,return_counts=True)):
                if count<2: continue
                loc=list(scene.cells[n])
                loc[free[0]],loc[free[1]]=map(int,pos)
                evidence[n][tuple(loc)]=evidence[n].get(tuple(loc),0)+int(count)
            proposals[(n,axis)]=True
    for n,owned in scene.owned.items():
        if len(evidence[n]) and all((n,int(a)) in proposals for a in np.unique(scene.axes[owned])):
            targets.pop(scene.cells[n],None)
    for n,votes in evidence.items():
        for loc in votes:
            targets.setdefault(loc,ids[scene.cells[n]])
    return targets,records,planes,models

def boundary(scene,planes,models,offsets,offered=None):
    # Compare neighboring faces at photographed seams. Every offered pixel already
    # belongs to one of the two ordinary planes. Background is left untouched.
    flat=planes.ravel()
    choices=flat.copy()
    best=np.zeros(len(flat),float)
    for axis in (0,1):
        pairs=np.argwhere((np.diff(planes,axis=axis)!=0))
        delta=scene.shape[1] if axis==0 else 1
        base=pairs[:,0]*scene.shape[1]+pairs[:,1]
        valid=(flat[base]>=0)&(flat[base+delta]>=0)
        a,b=base[valid],base[valid]+delta
        for owners,alternates in ((a,b),(b,a)):
            for k in np.unique(flat[owners]):
                sample=owners[flat[owners]==k]
                altidx=alternates[flat[owners]==k]
                owner=models[k]
                y,x=np.divmod(sample,scene.shape[1])
                xy=np.column_stack((x,y))*scene.step+.5
                gray=scene.rgb[y,x].mean(1)
                def predict(m,p):
                    c,axis,tile,gain,bias=m
                    return gain*tile[sample_uv(coordinates(scene,p[:,None]+offsets,c,axis))].mean(1)+bias
                old=np.abs(gray-predict(owner,xy))
                for rival in np.unique(flat[altidx]):
                    if models[rival][1]==owner[1] and models[rival][0][owner[1]]==owner[0][owner[1]]:
                        continue
                    use=flat[altidx]==rival
                    if offered is not None:offered.ravel()[sample[use]]=True
                    gain=old[use]-np.abs(gray[use]-predict(models[rival],xy[use]))
                    points=sample[use]
                    switch=gain>np.maximum(5,best[points])
                    choices[points[switch]]=rival
                    best[points[switch]]=gain[switch]
    return choices.reshape(planes.shape)

def run(row):
    ds,key=row['dataset'],row['id']
    path=OUT/'geometry-probes-final'/ds/(key+'.json')
    if path.exists():return read(path)
    rgb,camera,pack,assignment=inputs(row)
    local={p:v.split(':')[-1] for p,v in decoded_pack(BytesIO(pack)).items()}
    image=Image.open(rgb).convert('RGB')
    scene=Scene(image,[],camera,local,step=2)
    targets,records,planes,models=source(scene,local,_export_atlas(),OFFSETS)
    offered=np.zeros(scene.shape,bool)
    alt=boundary(scene,planes,models,OFFSETS,offered)
    one=np.zeros((1,2))
    static_targets,static_records,static_planes,static_models=source(scene,local,_export_atlas(),one)
    assert np.array_equal(planes,static_planes)
    static_alt=boundary(scene,planes,static_models,one)
    # Grader only, after both source-only selections are complete.
    entry=entries()[ds,key]
    frame=rgb.parent
    meta=read(frame/'labels.json')
    with np.load(frame/'labels.npz') as data:
        face=data['face_id'][::2,::2]
    map_id=np.zeros(max([f['id'] for f in meta['faces']],default=0)+1,int)
    for f in meta['faces']:map_id[f['id']]=f['surface_id']
    truth=map_id[face]
    # Bind each predicted plane to its dominant truth surface on the non-edited
    # interior. Reference cannot influence its shape, basis, sprite or decision.
    lut=np.zeros(len(models)+1,int)
    for k in range(len(models)):
        interior=(planes==k)&(alt==planes)&(static_alt==planes)
        labels=truth[interior]
        if len(labels):lut[k+1]=np.bincount(labels).argmax()
    changed=alt!=planes
    before=(lut[planes+1]==truth)&(truth>0)
    after=(lut[alt+1]==truth)&(truth>0)
    gains=int((changed&after&~before).sum())
    losses=int((changed&before&~after).sum())
    def edge_grade(grid_records):
        values=[]
        for rec in grid_records:
            k=rec['plane_number']
            ys,xs=np.nonzero(planes==k)
            choose=np.linspace(0,len(ys)-1,min(len(ys),3200),dtype=int)
            ys,xs=ys[choose],xs[choose]
            truth_ids=face[ys,xs]
            for dy,dx in ((0,1),(1,0)):
                y2,x2=np.minimum(ys+dy,scene.shape[0]-1),np.minimum(xs+dx,scene.shape[1]-1)
                ok=(truth[ys,xs]>0)&(truth[ys,xs]==truth[y2,x2])&(planes[y2,x2]==k)
                if not ok.any():continue
                xy=np.column_stack((xs,ys))[ok]*2+.5
                cell,axis,*_=models[k]
                uv=coordinates(scene,xy,cell,axis)-rec['start']
                uv2=coordinates(scene,xy+[2*dx,2*dy],cell,axis)-rec['start']
                native=(truth_ids!=face[y2,x2])[ok]
                for arm,scale,shift in [('base',1,0),('source',rec['scale'],rec['shift'])]:
                    pred=np.any(np.floor(uv*scale+shift)!=np.floor(uv2*scale+shift),axis=1)
                    values.append([arm,int((native&pred).sum()),int((native&~pred).sum()),int((~native&pred).sum())])
        return {a:np.sum([v[1:] for v in values if v[0]==a],axis=0).tolist() if values else [0]*3 for a in ['base','source']}
    static_pack,_=typed_pack(static_targets)
    static_row,_=evaluate_pack(entry,static_pack,camera,assignment={k:assignment.get(k,'source-atlas') for k in static_targets})
    static_after=(lut[static_alt+1]==truth)&(truth>0)
    static_change=static_alt!=planes
    p,decoded=typed_pack(targets)
    r,_=evaluate_pack(entry,p,camera,assignment={k:assignment.get(k,'source-atlas') for k in targets})
    res=dict(dataset=ds,id=key,split=row['split'],grid=records,source_grid_edges=edge_grade(records),
        point_grid=static_records,point_grid_edges=edge_grade(static_records),point_scores=static_row['scores'],
        point_boundary_switched=int(static_change.sum()),point_boundary_corrected=int((static_change&static_after&~before).sum()),
        point_boundary_damaged=int((static_change&before&~static_after).sum()),
        view_vs_point_corrected=int((after&~static_after).sum()),view_vs_point_damaged=int((static_after&~after).sum()),
        modeled_plane_samples=int((planes>=0).sum()),offered_boundary_samples=int(offered.sum()),boundary_switched=int(changed.sum()),boundary_corrected=gains,boundary_damaged=losses,
        boundary_same_correct=int((changed&before&after).sum()),boundary_both_wrong=int((changed&~before&~after).sum()),
        added=len(targets.keys()-local.keys()),removed=len(local.keys()-targets.keys()),baseline=row['scores'],source=r['scores'],
        baseline_geometry=row['geometry'],source_geometry=r['geometry'])
    save(path,res)
    return res

def main():
    rows=read(BASE/'scores/cases.json')
    # First pack-producing capture per dataset/layout, determined before scoring.
    grouped={}
    for r in rows:
        if r['dataset']!='fixed-104' and r['exported'] and r['camera_usable']:
            grouped.setdefault((r['dataset'],r['layout']),r)
    with ProcessPoolExecutor(5) as pool:
        cases=list(pool.map(run,grouped.values()))
    save(REVIEW/'geometry-probes.json',cases)

if __name__=='__main__':main()
