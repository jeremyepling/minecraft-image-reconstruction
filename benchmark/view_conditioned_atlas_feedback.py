"""Feed first-pass rendered face ownership back into existing pre-inset masks."""
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from unittest.mock import patch
import numpy as np
from PIL import Image
from benchmark.view_conditioned_atlas_geometry import source,boundary,OFFSETS,Scene
from benchmark.view_conditioned_atlas_run import BASE,OUT,REVIEW,inputs,entries,decoded_pack,read,save,_export_atlas
from benchmark.expanded_baseline import replay_source_fields
from benchmark.expanded_fixed_handoffs import suite
from benchmark.expanded_scores import score_build
from engine import inset_returns,surface_model
from engine.plane_consensus import row_runs
from engine.surface_boundary_refine import _retain_topology

def update(parts,old,new,image_size):
    sizes={tuple(part['support']['size']) for part in parts}
    if len(sizes)>1:
        updated=list(parts)
        total=0
        for size in sorted(sizes):
            indices=[i for i,p in enumerate(parts) if tuple(p['support']['size'])==size]
            subset,count=update([parts[i] for i in indices],old,new,image_size)
            for i,part in zip(indices,subset):updated[i]=part
            total+=count
        return updated,total
    size=tuple(parts[0]['support']['size'])
    original=np.zeros(size[::-1],np.int32)
    overlap=np.zeros(size[::-1],bool)
    for i,p in enumerate(parts,1):
        assert tuple(p['support']['size'])==size
        for y,a,b in p['support']['runs']:
            overlap[y,a:b] |= original[y,a:b]!=0
            original[y,a:b]=i
    y,x=np.indices(original.shape)
    sy=np.minimum((y*image_size[1]/size[1]/2).astype(int),old.shape[0]-1)
    sx=np.minimum((x*image_size[0]/size[0]/2).astype(int),old.shape[1]-1)
    src,dest=old[sy,sx],new[sy,sx]
    lut=np.zeros(old.max()+2,int)
    for n in np.unique(src[src>=0]):
        ids=original[(src==n)&(src==dest)&~overlap]
        if len(ids):
            hits=np.bincount(ids)
            if hits.max()>=.55*len(ids):
                lut[n+1]=hits.argmax()
    target=lut[dest+1]
    move=(src!=dest)&(original!=0)&(original==lut[src+1])&(target!=0)&~overlap
    result=_retain_topology(original,np.where(move,target,original))
    y,x=np.nonzero(result!=original)
    touched=set(original[y,x])|set(result[y,x])
    # Retain native overlapping supports and all other unaffected mask pixels.
    changed=[]
    for i,p in enumerate(parts,1):
        if i in touched:
            mask=np.zeros(original.shape,bool)
            for yy,a,b in p['support']['runs']:mask[yy,a:b]=True
            mask[y,x]=result[y,x]==i
            p=dict(p,support=dict(size=list(size),runs=row_runs(mask)),support_area=int(mask.sum()))
        changed.append(p)
    return changed,len(x)

def replay(row):
    ds,key=row['dataset'],row['id']
    entry=entries()[ds,key]
    rgb,camera,pack,assignment=inputs(row)
    local={p:v.split(':')[-1] for p,v in decoded_pack(BytesIO(pack)).items()}
    image=Image.open(rgb).convert('RGB')
    scene=Scene(image,[],camera,local,step=2)
    _,_,planes,models=source(scene,local,_export_atlas(),OFFSETS)
    changed=boundary(scene,planes,models,OFFSETS)
    original=inset_returns.detect
    outputs={}
    for arm in ['noop','source']:
        folder=OUT/'mask-feedback'/key/arm
        folder.mkdir(parents=True,exist_ok=True)
        count=[]
        def hook(image,parts,*args,**kwargs):
            if arm=='source':
                updated,n=update(parts,planes,changed,image.size)
                parts[:]=updated
                count.append(n)
            return original(image,parts,*args,**kwargs)
        with replay_source_fields(rgb,BASE/ds/key),patch.object(inset_returns,'detect',hook):
            surface_model.build(rgb,'26.3',folder,through='placement',atlas_path=suite.ATLAS)
        result=score_build(entry,folder)
        pred=decoded_pack(folder/'reconstruction.zip')
        outputs[arm]=dict(scores=result['scores'],geometry=result['geometry'],mask_samples_changed=sum(count),
                          ids_changed=sum(pred[p]!=v for p,v in decoded_pack(BytesIO(pack)).items() if p in pred),
                          pack_sha256=result['pack_sha256'],
                          tie=pred==decoded_pack(BytesIO(pack)),
                          counts=[(r['id'],r.get('counts'),r.get('phase')) for r in read(folder/'counts.json')['surfaces']])
    return dict(dataset=ds,id=key,baseline=row['scores'],**outputs)

def main():
    keys=['train-market-court-1-warm-high-right','test-walled-garden-1-warm-low-left','validation-hillside-chapel-1-warm-low-left']
    jobs=[r for r in read(BASE/'scores/cases.json') if r['id'] in keys]
    with ProcessPoolExecutor(3) as pool:
        rows=list(pool.map(replay,jobs))
    save(REVIEW/'mask-feedback.json',rows)
if __name__=='__main__':main()
