"""Score the same atlas lookup on photographed and predicted single-block footprints."""
from collections import Counter,defaultdict
import hashlib

import numpy as np
from PIL import Image

from benchmark.camera_round import read,save
from benchmark.count_metrics import reference_planes
from benchmark.surface_reference import recorded
from benchmark.textures import match_cells
from engine.surface_tiles import tile_sample,visible_tiles


def identity(value):
    return value.removeprefix('minecraft:') if value else None


def ranked(sample, atlas):
    from engine.atlas_count import rank_tile
    outcome=rank_tile(sample,atlas,version='26.3')
    return dict(block_id=identity(outcome.get('block_id')),texture=outcome.get('texture'),
                ranked_ids=[identity(n) for n in outcome.get('ranked_ids',[])][:5])


def true_faces(planes,size):
    valid={}
    for pid,plane in planes.items():
        for cell,f in plane['faces'].items():
            q=np.asarray(f['image_corners'])
            if (f['visible_fraction']<1.-1e-9 or not all(f['corner_visible'])
                or (q<0).any() or (q>size).any() or np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).min()<4.):
                continue
            valid[pid,cell]=f
    return valid


def shape_key(q):
    return np.asarray(q,dtype=np.float64).tobytes()


def picture(result,row,case,atlas,frozen,method):
    from benchmark.count_round import cached_regions
    from engine.surface_sample import sample_surface
    with Image.open(case.path('image')) as source:
        image=source.convert('RGB')
    tiles=[]
    by_proposal=defaultdict(list)
    for part,rec in zip(cached_regions(row,frozen),result['surfaces'],strict=True):
        sample=sample_surface(image,part)
        for tile in visible_tiles(sample,rec['measures'][method]):
            tile_id=len(tiles)
            by_proposal[part['id']].append(tile_id)
            tiles.append(dict(proposal=part['id'],cell=tile['cell'],quad=tile['quad'],**ranked(tile,atlas)))
    # Scoring only. The IDs and quads above were fixed before accessing any known face.
    _,ref=recorded(case)
    valid=true_faces(reference_planes(ref,read(case.path('labels'))['faces']),image.size)
    linked={}
    for rec in result['surfaces']:
        pid=rec['recorded_surface']
        if pid is None:
            continue
        tile_ids=by_proposal[rec['id']]
        shapes=[tiles[i]['quad'] for i in tile_ids]
        indices={shape_key(q):i for q,i in zip(shapes,tile_ids)}
        eligible={cell for p,cell in valid if p==pid}
        matched,_=match_cells(shapes,eligible,np.linalg.inv(ref['charts'][pid]))
        for cell,(q,iou,purity) in matched.items():
            idx=indices[shape_key(q)]
            linked[pid,cell]=idx
            f=valid[pid,cell]
            tiles[idx].update(truth=identity(f['block_id']),xyz=f['xyz'],direction=f['direction'],iou=iou,purity=purity)
    # Fixed, prediction-independent sample, plus the exact faces matched above.
    def order(key):
        f=valid[key]
        return hashlib.sha256(str((row['sha256'],f['xyz'],f['direction'])).encode()).digest()
    fixed=sorted(valid,key=order)[:12]
    known=[]
    for key in dict.fromkeys([*fixed,*linked]):
        f=valid[key]
        known.append(dict(xyz=f['xyz'],direction=f['direction'],truth=identity(f['block_id']),
            quad=f['image_corners'],fixed_sample=key in fixed,paired_tile=linked.get(key),
            **ranked(tile_sample(image,f['image_corners']),atlas)))
    counts=Counter(eligible_faces=len(valid),proposed=len(tiles),matched=len(linked),fixed_faces=len(fixed),
                   fixed_top1=0,fixed_top5=0,paired_known_top1=0,paired_known_top5=0,predicted_top1=0,predicted_top5=0)
    for item in known:
        for use,key in ((item['fixed_sample'],'fixed'),(item['paired_tile'] is not None,'paired_known')):
            if use:
                counts[key+'_top1']+=int(item['truth']==item['block_id'])
                counts[key+'_top5']+=int(item['truth'] in item['ranked_ids'])
    for i in linked.values():
        tile=tiles[i]
        counts['predicted_top1']+=int(tile['truth']==tile['block_id'])
        counts['predicted_top5']+=int(tile['truth'] in tile['ranked_ids'])
    return dict(key=row['key'],counts=dict(counts),known=known,predicted=tiles)


def evaluate_materials(results,source,lookup,atlas,frozen,method,out):
    inputs={r['key']:r for r in source}
    records=[]
    totals=Counter()
    for result in results:
        key=result['key']
        row=picture(result,inputs[key],lookup[key],atlas,frozen,method)
        records.append(row)
        totals.update(row['counts'])
    save(out/'materials.json',dict(counter=method,matcher='atlas single-face lookup',all=dict(totals),
        protocol='For each frame, up to 12 independently hash-selected fully visible faces at least four pixels on each edge. Also compare true versus predicted corners on every recovered eligible face (plane match, cell overlap >= 0.5).',pictures=records))
