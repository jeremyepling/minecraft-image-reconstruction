"""Second source-only grouping test: containment of touching partial fragments in one projected cube."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import argparse
import numpy as np
from benchmark import full_block_lattice_run as run
from benchmark import full_block_lattice_groups as groups
from benchmark.expanded_baseline import frames,SOURCE,read,save
from benchmark.full_block_lattice import rays,line_camera
from benchmark.mask_blocks import decoded_pack
from benchmark.typed_structures import typed_pack


def cube_coverage(cell, fragment, eye):
    lower=np.array(cell,dtype=float)
    # Half a texel accommodates sample-center discretization. Test every retained visible ray.
    with np.errstate(divide='ignore',invalid='ignore'):
        one=(lower-eye-.03125)/fragment['rays']
        two=(lower-eye+1.03125)/fragment['rays']
    inside=(np.minimum(one,two).max(1)<=np.maximum(one,two).min(1)) & (np.maximum(one,two).min(1)>0)
    plane_error=min(max(lower[a]-level,level-lower[a]-1,0.) for a,level in fragment['planes'])
    return float(inside.mean())-.45*plane_error


def decisions(document,P,eye,locked,width):
    pieces=document['fragments']
    enriched=[]
    for f in pieces:
        points=np.array(f['native_pixel_indices'],dtype=int)
        x,y=points%width,points//width
        xy=np.column_stack((x,y))+.5
        enriched.append(dict(**f,points=set(points.tolist()),rays=rays(P[:,:3],xy),
            halo={int(p+d) for p in points for d in (4,-4,4*width,-4*width)}))
    by_cell=defaultdict(list)
    for i,frag in enumerate(enriched):
        by_cell[tuple(frag['local'])].append(i)
    options=[]
    for cell,ii in by_cell.items():
        if cell in locked or len(ii)!=1:
            continue
        one=enriched[ii[0]]
        for delta in product((-1,0,1),repeat=3):
            neighbor=tuple(a+b for a,b in zip(cell,delta))
            jj=by_cell.get(neighbor)
            if neighbor<=cell or neighbor in locked or not jj or len(jj)!=1 or sum(abs(x) for x in delta)>2:
                continue
            two=enriched[jj[0]]
            if (one['surface']==two['surface'] or one['planes'][0][0]==two['planes'][0][0]
                    or not one['points'] or not two['points']):
                continue
            subcell=max(abs(v-round(v)) for f in (one,two) for _,v in f['planes'])
            if subcell<.125:
                continue
            join=len(one['halo'] & two['points'])
            if join<2:
                continue
            scores=[min(cube_coverage(c,f,eye) for f in (one,two)) for c in (cell,neighbor)]
            best=int(np.argmax(scores))
            if scores[best]<.91:
                continue
            keep,remove=(cell,neighbor) if best==0 else (neighbor,cell)
            options.append((scores[best],join,keep,remove))
    available=set()
    edits={}
    for score,join,keep,remove in sorted(options,reverse=True):
        if keep in available or remove in available:
            continue
        available.update((keep,remove))
        edits[remove]=keep
    return by_cell,edits


def one(entry):
    root=run.BASE/entry['dataset']/entry['id']
    out=run.RUN/entry['dataset']/entry['id']
    evidence=read(out/'evidence.json.gz')
    row=read(out/'result.json')
    document=read(out/'grouped-observations.json.gz')
    anchors=evidence.get('anchors',[])
    before=read(out/'grouping.json')
    changes={}
    known=read(root/'build-manifest.json')['placement']
    camera=known.copy()
    if anchors:
        eye=np.array(evidence['eye'])
        if known['P'] is not None:
            P=np.array(known['P'])
        else:
            lc=line_camera(read(root/'handoffs/mlsd.json.gz')['segments'],(1440,900))
            P=np.column_stack((lc,-lc@eye))
        locked=set()
        for a in anchors:
            ax=a['axis']
            center=np.array(a['xyz']).mean(0)+eye
            center[ax]-=np.sign(eye[ax]-center[ax])*.02
            locked.add(tuple(np.floor(center+1e-7).astype(int).tolist()))
        by_cell,changes=decisions(document,P,eye,locked,1440)
        camera.update(P=P.tolist(),camera_status='supported')
    else:
        by_cell={}
    membership=defaultdict(list)
    for p,items in by_cell.items():
        membership[changes.get(p,p)].extend(items)
    # Ordinary occupied coordinates retain their existing placement. Only uncorroborated
    # new occupied cells can be removed. A merged destination already had source evidence.
    baseline=decoded_pack(root/'reconstruction.zip')
    proposal=decoded_pack(out/'proposal.zip')
    remove=set(changes)-baseline.keys() if row['changed'] else set()
    removed_from_pack={p for p in remove if p in proposal}
    payload=typed_pack({p:name.removeprefix('minecraft:') for p,name in proposal.items() if p not in remove})[0]
    if not row['changed']:
        payload=(root/'reconstruction.zip').read_bytes()
        camera=known
    (out/'containment-source.zip').write_bytes(payload)
    # Reference boundary: after relationships and real export are frozen.
    if removed_from_pack:
        methods={tuple(b[a] for a in 'xyz'):b['material_source'] for b in read(root/'placements.json')['blocks']}
        for p,id_ in decoded_pack(out/'containment-source.zip').items():
            methods.setdefault(p,'fallback' if id_=='minecraft:smooth_stone' else 'source-atlas')
        scored,_=run.scorer.evaluate_pack(entry,payload,camera,assignment=methods)
        scored=run.compact_row(scored)
    else:
        scored=row['source']
    labels=read(Path(entry['frame'])/'labels.json')
    with np.load(Path(entry['frame'])/'labels.npz') as native:
        raster=native['face_id']
    graded=groups.grade([np.array(f['native_pixel_indices'],dtype=int) for f in document['fragments']],
                        list(membership.values()),raster,labels)
    result=dict(id=entry['id'],split=entry['split'],edits=[[a,b] for a,b in changes.items()],
                removed_export=len(removed_from_pack),before=before['grouped'],after=graded,source=scored)
    save(out/'containment.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        rows=list(pool.map(one,list(frames(SOURCE,['decorative-builds-v1']))))
    def totals(rows):
        before,after=Counter(),Counter()
        for r in rows:
            before.update(r['before']); after.update(r['after'])
        return dict(tested=len(rows),before=dict(before),after=dict(after),
                    proposed_merges=sum(len(r['edits']) for r in rows),removed_export=sum(r['removed_export'] for r in rows),
                    scored={k:run.typed.pooled_scores([r['source'][k] for r in rows]) for k in ('geometry','scores')})
    save(run.REVIEW/'containment.json',dict(all=totals(rows),splits={s:totals([r for r in rows if r['split']==s]) for s in {r['split'] for r in rows}}))
    save(run.REVIEW/'containment-cases.json.gz',rows)


if __name__=='__main__':
    main()
