"""Grade frozen development decisions with visible cell-height labels after inference."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as ev
from engine.visible_horizontal_faces import _surface_masks


def run(roots,out):
    records={p.name:{r['id']:r for r in ev.read(p/'cases.json')} for p in roots}
    summary={name:[] for name in records}
    for split,f,prior in ev.cases(['train']):
        key=f['id']
        image,parts,counts,camera=ev.photo(split,f,prior)
        _,_,T,_=ev.references(split,f,None,ev.TRAINING_DATA,ev.DEPTH,{})
        source=ev.TRAINING_DATA/'data'/split/f['directory']
        label=ev.read(ev.checked(source/'labels.json',f['files']['labels.json']))
        with np.load(ev.checked(source/'labels.npz',f['files']['labels.npz'])) as data:
            ids=data['face_id'][::2,::2].copy()
        assert T[1,1]==1 and not T[1,0] and not T[1,2]
        offsets=np.full(max(face['id'] for face in label['faces'])+1,-999,int)
        horiz=np.zeros(len(offsets),bool)
        for face in label['faces']:
            i=face['id']
            offsets[i]=int(np.rint(face['xyz_world'][1]-T[1,3]))
            horiz[i]=bool(face['normal_world'][1])
        hist={}
        for part in parts:
            if not part.get('support'):
                continue
            mask=_surface_masks([part],image.size,np.arange(image.width)[::2]+.5,np.arange(image.height)[::2]+.5,np.eye(3))[0]
            valid=ids[mask]
            for rim in (True,False):
                owned=offsets[valid[(valid>0)&(horiz[valid]!=rim)]]
                top=Counter(owned.tolist()).most_common(1)
                hist[part['id'],rim]=top[0][0] if top and top[0][1]>.5*len(valid) else None
        for name,outputs in records.items():
            output=outputs[key]
            primary=output['primary_height']
            tested=[]
            for d in output['height_decisions']:
                reference=hist[d['part'],d.get('kind')=='one_course_rim']
                tested.append(dict(part=d['part'],block_y=d['block_y'],accepted=d['accepted'],reference=reference))
            summary[name].append(dict(id=key,rgb_sha256=f['files']['rgb.png'],decisions=tested,
                                      primary=next((d for d in tested if primary and d['part']==primary['part']),None),
                                      correct=output['correct'],extras=output['extra'],hidden=output['hidden']))
    out.mkdir(parents=True,exist_ok=True)
    ev.save(out/'development.json',summary)
    for k,rows in summary.items():
        all_picks=[d for r in rows for d in r['decisions']]
        for cohort,picks in [('accepted',[d for d in all_picks if d['accepted']]),('primary',[r['primary'] for r in rows if r['primary']])]:
            print(k,cohort,len(picks),'correct',sum(d['block_y']==d['reference'] for d in picks),'no majority',sum(d['reference'] is None for d in picks))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run(args.runs,args.out)
