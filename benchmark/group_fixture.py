"""Prepare fully observed predicted and diagnostic faces; keep scored block identities in separate files."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine.geometry import transform, UNIT, project
from engine.surface_sample import sample_surface
from engine.surface_tiles import complete_footprints
from engine.masked_count import count
from benchmark.textures import area, intersection

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'benchmark/reviews/crop-split-01'
def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data): path.write_text(json.dumps(data,separators=(',',':')))
def observed(task):
    case,image,mask_hash=task
    mask=BASE/case['predictions']
    assert sha(mask)==mask_hash and sha(image)==case['sha256']
    surfaces=json.loads(gzip.decompress(mask.read_bytes()))['split']
    with Image.open(image) as im:
        rgb=im.convert('RGB')
    rows=[]
    for surface in surfaces:
        sample=sample_surface(rgb,surface)
        for cell,q in complete_footprints(sample,count(sample)):
            rows.append(dict(id=len(rows),quad=q.tolist(),proposal=surface['id'],normal_camera=surface.get('normal_camera'),cell=cell))
    return dict(kind='predicted',key=case['key'],image=str(image),sha256=case['sha256'],mask_sha256=mask_hash,
                width=rgb.width,height=rgb.height,surfaces=len(surfaces),faces=rows)

def label(frame, labels):
    records=read(labels)['faces']
    refs=[]
    for r in records:
        q=np.array(r['image_corners'])
        if not all(r['corner_visible']) or r['visible_fraction']<1-1e-9 or (q<0).any() or (q>[frame['width'],frame['height']]).any() or area(q)<.5:
            continue
        refs.append(r)
    # IoU in the reference face plane, as used by the count-01 evaluator.
    candidates=[]
    for r in refs:
        q=np.array(r['image_corners']); lo=q.min(0);hi=q.max(0)
        for row in frame['faces']:
            observed=np.array(row['quad'])
            if np.any(observed.min(0)>hi) or np.any(observed.max(0)<lo): continue
            uv=project(np.linalg.inv(transform(q)),observed)
            if not np.isfinite(uv).all(): continue
            a=area(uv)
            if not .5<=a<=2:continue
            common=intersection(uv,(0,0))
            iou=common/(1+a-common)
            if iou>=.5: candidates.append((iou,row['id'],r))
    seen=set(); taken=set(); truth=[]
    for iou,idx,r in sorted(candidates,key=lambda t:-t[0]):
        identifier=(tuple(r['xyz']),r['direction'])
        if idx in seen or identifier in taken:continue
        seen.add(idx);taken.add(identifier)
        truth.append(dict(id=idx,xyz=r['xyz'],direction=r['direction'],block_id=r['block_id'].removeprefix('minecraft:'),iou=iou))
    return refs,dict(key=frame['key'],kind='predicted',faces=truth,label_sha256=sha(labels))

def diagnostic(frame,refs):
    # Select one contiguous review area around a reproducibly chosen visible block corner.
    # The centered view intentionally enriches shared-block edges and includes all nearby negatives.
    groups=defaultdict(list)
    for r in refs: groups[tuple(r['xyz'])].append(r)
    shared=[pos for pos,rs in groups.items() if len(rs)>1]
    raw_id=lambda pos:hashlib.sha256(str((frame['key'],pos)).encode()).digest()
    anchor=min(shared or groups,key=raw_id) if groups else None
    centers={pos:np.mean([r['image_corners'] for r in rs],axis=(0,1)) for pos,rs in groups.items()}
    selected=sorted(groups,key=lambda pos:(np.linalg.norm(centers[pos]-centers[anchor]),pos))[:12]
    obs=[];targets=[]
    # Order by image position only; order carries no source block grouping.
    chosen=[r for pos in selected for r in groups[pos]]
    chosen.sort(key=lambda r:tuple(np.mean(r['image_corners'],axis=0)))
    for i,r in enumerate(chosen):
        obs.append(dict(id=i,quad=r['image_corners'],proposal=None,cell=None,normal_camera=None))
        targets.append(dict(id=i,xyz=r['xyz'],direction=r['direction'],block_id=r['block_id'].removeprefix('minecraft:')))
    return dict(frame,kind='recorded',faces=obs),dict(key=frame['key'],kind='recorded',faces=targets)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=read(BASE/'manifest.json')
    m=ROOT/'benchmark/reviews/mask-four-01'
    sources={r['key']:(m/r['source']).resolve() for r in read(m/'cases.json')}
    labels={}
    for idx in (ROOT/'benchmark/data').glob('*/index.json'):
        for r in read(idx).get('frames',[]):
            if 'labels' in r:labels[(idx.parent,r['image_sha256'])]=idx.parent/r['labels']
    tasks=[(r,sources[r['key']],meta['artifacts'][r['predictions']]) for r in read(BASE/'cases.json')]
    with ThreadPoolExecutor(4) as pool: frames=list(pool.map(observed,tasks))
    save(args.out/'faces.json',dict(frames=frames))
    diag=[];ds=[];truth=[]
    for f in frames:
        refs,t=label(f,labels[Path(f['image']).parent.parent,f['sha256']]);truth.append(t)
        a,b=diagnostic(f,refs);diag.append(a);ds.append(b)
    for name,records in [('truth',truth),('diagnostic-faces',diag),('diagnostic-truth',ds)]:
        save(args.out/(name+'.json'),dict(frames=records))
    save(args.out/'input-digests.json',{name:sha(args.out/name) for name in ('faces.json','truth.json','diagnostic-faces.json','diagnostic-truth.json')})
