"""Reference-only coverage and clean attainable block oracle for the three strip orientations."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed, wall_section_eval
from benchmark import strip_face_directions as candidate
from benchmark.camera_metrics import registration
from benchmark.facade_visibility import square_support
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked,read,save
from benchmark.visible_error_budget import projection,stage_evidence,tally

CONTROL=Path('../visible-error-budget/runs/primary-production')
MODES=('conservative','horizontal','corroborated')


def score(job):
    split,fr,h,row=job
    key=fr['id']
    image,parts,counts,camera=framed.photo(split,fr,h)
    cells=set(decoded_pack(checked(CONTROL/split/key/'reconstruction.zip',row['pack_sha256'])))
    scene,patches,stats=candidate.analyze(image,parts,camera,cells)
    # Reference geometry and recorded face orientations are opened only for scoring.
    _,_,counts,camera,pose,labels,ids,truth,_,placed,_=read_view(CONTROL,split,fr,h,row)
    owner,_,world_axis,_=projection(camera,pose,sorted(cells),row['translation'],ids.shape)
    T=registration(scene.P,pose)
    def world(c):
        return tuple(np.floor((np.array(c)+.5)@T[:3,:3].T+T[:3,3]).astype(int)+row['translation'])
    blocks={b['id']:tuple(b['xyz_world']) for b in labels['blocks']}
    n=len(blocks)+1
    block=np.zeros(len(labels['faces'])+1,int)
    axis=block.copy()
    label=block.copy()
    roles=defaultdict(set)
    for f in labels['faces']:
        norm=f['normal_world']
        face=f['id'];b=f['block']
        role=2 if norm[1]>0 else 3 if norm[1]<0 else 1
        axis[face],label[face],block[face]=np.argmax(abs(np.array(norm))),role,b
        roles[b].add(role)
    footprint=block[ids]
    area=np.bincount(footprint.ravel(),minlength=n)
    ranking=stage_evidence(parts,counts,image.size,camera,world_axis,axis[ids])
    any_mask=tally(footprint.ravel(),(ranking>0).ravel(),n)*2>=area
    same_mask=tally(footprint.ravel(),(ranking>1).ravel(),n)*2>=area
    uncovered=tally(footprint.ravel(),(owner>=0).ravel(),n)*2<area
    targets={b for b,q in blocks.items() if q not in placed and 1 not in roles[b] and
             any_mask[b] and not same_mask[b] and uncovered[b]}
    square=square_support(footprint,n-1)
    optional={q for b,q in blocks.items() if 1 not in roles[b] and square[b]<4}
    cohort={blocks[b] for b in targets}
    samples=ids[::candidate.STEP,::candidate.STEP].ravel()
    scores=Counter(targets=len(targets),target_required=len(cohort-optional))
    confusion=Counter()
    pool=defaultdict(lambda:defaultdict(set))
    touched=set(); offered=set()
    for pixels,upstream,opts,winner in patches:
        source=samples[pixels]
        touched.update(set(block[source]) & targets)
        kinds={candidate.interpret(o) for o in opts}
        if 'wall' in kinds and len(kinds)>1:
            c=np.bincount(label[source],minlength=4)
            if c.argmax() and c.max()*2>len(source):
                reference=('background','wall','top','underside')[c.argmax()]
                decision=candidate.interpret(winner)
                confusion[reference+'_'+decision]+=1
        for opt in opts:
            screenaxis=world_axis[opt['axis']]
            accepted_ids=set(block[source[axis[source]==screenaxis]])
            pos={blocks[b] for b in accepted_ids if b and blocks[b] not in placed}
            for cell in opt['cells']:
                dest=world(cell)
                if dest in pos:
                    pool[cell][opt['axis']].update(pixels.tolist())
                    offered.add(dest)
    scene.size=image.size
    oracle={world(c) for c in candidate.materialize(scene,pool)}
    scores.update(touched=len(touched),touched_required=sum(blocks[b] not in optional for b in touched),
                  offered=len(offered),offered_target=len(offered&cohort),offered_required=len((offered&cohort)-optional),
                  oracle=len(oracle),oracle_target=len(oracle&cohort),oracle_nonoptional=len(oracle-optional),
                  oracle_target_required=len((oracle&cohort)-optional))
    for method in MODES:
        root=Path('runs')/method
        result=set(decoded_pack(root/split/key/'reconstruction.zip'))
        guessed={world(c) for c in result-cells}
        scores[method+'_target']=len(guessed&cohort)
        scores[method+'_target_required']=len((guessed&cohort)-optional)
        scores[method+'_required_gained']=len((guessed&truth)-optional)
        scores[method+'_optional_gained']=len(guessed&optional)
    return dict(id=key,split=split,rgb_sha256=row['rgb_sha256'],pack_sha256=row['pack_sha256'],
                counts=scores,confusion=confusion,patches=stats)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    control=read(CONTROL/'summary.json')
    index={(r['split'],r['id']):r for r in read(checked(CONTROL/'cases.json',control['cases_sha256']))}
    jobs=[(s,f,h,index[s,f['id']]) for s,f,h in wall_section_eval.cases(framed.EVALUATION)]
    with ProcessPoolExecutor(args.workers) as workers:
        rows=list(workers.map(score,jobs))
    totals={field:dict(sum((Counter(r[field]) for r in rows),Counter())) for field in ('counts','patches','confusion')}
    assert totals['counts']['targets']==632 and totals['counts']['target_required']==17
    print(totals)
    save(args.out,dict(control_cases_sha256=control['cases_sha256'],totals=totals,frames=rows))
