"""Measure how much a source-only wall partition could recover with reference depth per patch."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.harbor_carrier_audit import BASE, CASES, REVIEW, SOURCE, build_dir, frames, native_face_planes, read, save
from benchmark.harbor_plane_replay import sampled_ids
from engine.planarity_crops import registered_tiles
from engine.visible_raster import observed_support


def bounded_rgb_regions(image, mask, tolerance=20, minimum=16):
    """Connect neighboring 4x4 samples with similar median source RGB; small regions share the residual."""
    h,w = mask.shape
    rgb=np.asarray(image)[:h*4,:w*4].reshape(h,4,w,4,3)
    rgb=np.median(rgb,axis=(1,3))
    horizontal=np.abs(rgb[:,1:] - rgb[:,:-1]).max(axis=2)<=tolerance
    vertical=np.abs(rgb[1:] - rgb[:-1]).max(axis=2)<=tolerance
    remaining=mask.copy()
    result=np.zeros(mask.shape,int)
    group=0
    for y,x in zip(*np.nonzero(mask)):
        if not remaining[y,x]:
            continue
        pending=[(y,x)]
        remaining[y,x]=False
        region=[]
        while pending:
            yy,xx=pending.pop()
            region.append((yy,xx))
            for dy,dx,open_edge in ((0,-1,xx>0 and horizontal[yy,xx-1]),(0,1,xx+1<w and horizontal[yy,xx]),
                                    (-1,0,yy>0 and vertical[yy-1,xx]),(1,0,yy+1<h and vertical[yy,xx])):
                j,i=yy+dy,xx+dx
                if open_edge and remaining[j,i]:
                    pending.append((j,i))
                    remaining[j,i]=False
        if len(region)>=minimum:
            group+=1
            rows,cols=zip(*region)
            result[rows,cols]=group
    return result


def audit(entry,case):
    key=entry['id']
    folder=build_dir(entry,case)
    frame=Path(entry['frame'])
    tag=read(REVIEW/(key+'.json'))['carriers'][0]['id']
    image=Image.open(frame/'rgb.png').convert('RGB')
    labels=read(frame/'labels.json')
    native=sampled_ids(labels,0)
    planes=native_face_planes(labels,np.array(case['registration']['transform']))
    reference=np.array([[-1,0]]+[planes[i] for i in range(1,len(planes)+1)])[native]
    selected=next(p for p in read(folder/'build-manifest.json')['placement']['plane_offsets'] if p['id']==tag)
    part=next(p for p in read(folder/'surfaces.json')['planes'] if p['id']==tag)
    support=observed_support(part['support'],native.shape,image.size)
    axis,level=selected['axis'],selected['level']
    same=support & (reference[:,:,0]==axis)
    levelmap=reference[:,:,1]

    def score(grouping):
        right=0
        bounded=0
        patches=[]
        for j in np.unique(grouping[support]):
            region=support & (grouping==j)
            levels,n=np.unique(levelmap[region&same],return_counts=True)
            target=int(levels[n.argmax()]) if len(n) and j else level
            right+=int((region & same & (levelmap==target)).sum())
            within=(abs(levels-level)<=6) if j else (levels==level)
            bounded+=int(n[within].max()) if within.any() else 0
            patches.append(dict(n=int(region.sum()),oracle=target))
        return dict(patches=len(patches)-int(0 in grouping[support]),correct=right,
                    correct_within_primary_six=bounded)

    whole=int(np.unique(levelmap[same],return_counts=True)[1].max())
    options={'RGB-connected':score(bounded_rgb_regions(image,support))}
    with np.load(BASE/entry['dataset']/key/'handoffs/planarity.npz') as planarity:
        options['planarity-global']=score(planarity['relaxed'])
        for i,canvas,_,_ in registered_tiles(planarity):
            options[f'planarity-tile-{i}']=score(canvas)
    result=dict(id=key,tag=tag,primary=level,axis=axis,same_axis_samples=int(same.sum()),all_samples=int(support.sum()),
                current_correct=int((same&(levelmap==level)).sum()),best_single=whole,partitions=options)
    save(REVIEW/(key+'-partition-probe.json'),result)
    print(key,result['same_axis_samples'],result['current_correct'],whole,options['RGB-connected'],
        options['planarity-global'],max((v['correct'],k) for k,v in options.items() if 'tile' in k),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids',nargs='+')
    args=p.parse_args()
    cases={r['id']:r for r in read(CASES)}
    for e in frames(SOURCE,['visible-builds-v3']):
        if e['id'] in args.ids:
            audit(e,cases[e['id']])
