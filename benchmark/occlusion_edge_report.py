"""Grade frozen proposals and compare actual exported inventories under each policy."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from benchmark.expanded_baseline import ROOT, STABLE, digest, read, save
from benchmark.harbor_carrier_audit import native_face_planes
from benchmark.occlusion_edge_run import CONTROL, OUT
from benchmark.typed_structures import pooled_scores

REVIEW = ROOT/'benchmark/reviews/occlusion-edge-depth-01'
ZERO = dict(gained=0,lost=0,extras_added=0,extras_removed=0)


def reference_levels(frame, plan):
    if frame['dataset']=='fixed-104' or not plan.get('audit',{}).get('offers'):
        return
    reference=frame['reference']
    def checked(name):
        path=STABLE.parent/reference[name]['path']
        assert digest(path)==reference[name]['sha256']
        return path
    labels=read(checked('labels.json'))
    with np.load(checked('labels.npz')) as data:
        # Native pixel nearest the placer's nominal four-pixel center.
        ids=data['face_id'][2::4,2::4]
    transform=np.array(frame['baseline']['registration']['transform'])
    lut=native_face_planes(labels,transform)
    axis,level=np.array([[-1,0]]+[lut[i] for i in range(1,len(lut)+1)])[ids].transpose(2,0,1)
    with np.load(OUT/frame['id']/'labels.npz') as cache:
        regions=cache['labels']
    for o in plan['audit']['offers']:
        def frequency(group):
            return Counter(map(int,level[(regions==group)&(axis==plan['axis'])]))
        freq=frequency(o['back'])
        near=frequency(o['front'])
        feasible=[r['level'] for r in o['rows'] if r['behind']]
        o['reference']=dict(same_axis=sum(freq.values()),levels=dict(freq),incumbent=freq[plan['initial']],
            selected=freq[o['level']],best_ordered=max((freq[l] for l in feasible),default=0),
            best_raw=max((freq[r['level']] for r in o['rows']),default=0),
            front_samples=sum(near.values()),front_correct=near[o['anchor']])


def overlay(records):
    keys=['train-amphitheater-02-ashlar-high-right','validation-harbor-02-kiln-high-right','validation-harbor-03-seafoam-high-right']
    canvas=Image.new('RGB',(1260,340),'#132029')
    pen=ImageDraw.Draw(canvas)
    for col,key in enumerate(keys):
        r=next(r for r in records if r['id']==key)
        o=r['plan']['audit']['offers'][0]
        with Image.open(STABLE.parent/r['rgb']['path']) as im:
            image=np.asarray(im.convert('RGB')).copy()
        with np.load(OUT/key/'labels.npz') as cache:
            labels=cache['labels'].repeat(4,axis=0).repeat(4,axis=1)
        for group,color in ((o['back'],[0,218,255]),(o['front'],[255,166,48])):
            selected=labels==group
            image[selected]=image[selected]*.47+np.array(color)*.53
        image=Image.fromarray(image)
        draw=ImageDraw.Draw(image)
        for y,x in o['points']:
            draw.ellipse((4*x-10,4*y-10,4*x+10,4*y+10),outline='white',width=4)
        image.thumbnail((420,245))
        x=420*col
        canvas.paste(image,(x,0))
        pen.text((x+6,249),key.replace('validation-','').replace('train-',''),fill='white',font_size=15)
        pen.text((x+6,272),f"Rear level {r['plan']['initial']} to {o['level']}   |   "+('accepted' if o['eligible'] else 'forced alternate'),fill='white',font_size=15)
    pen.text((7,313),'Orange marks the proposed foreground cap, cyan the rear region, and white the source junction.',fill='white',font_size=17)
    canvas.save(REVIEW/'source-junctions.jpg',quality=93)


def pool(rows,policy):
    selected=[r[r['choice'][policy]] for r in rows]
    return dict(views=len(rows),camera_usable=len(rows),geometry=pooled_scores([r['geometry'] for r in selected]),
                scores=pooled_scores([r['scores'] for r in selected]),
                paired={k:sum(r['paired'][k] for r in selected) for k in ZERO})


def main():
    controls=read(CONTROL)
    records=[]
    for frame in controls['pilot']['frames']:
        key=frame['id']
        plan=read(OUT/key/'plan.json')
        assert plan['source_rule_sha256']==digest(ROOT/'benchmark/occlusion_edge_depth.py')
        for offer in plan.get('audit',{}).get('offers',[]):
            levels = {r['level']: r for r in offer['rows']}
            best, prior = levels[offer['level']], levels[plan['initial']]
            failed = [reason for passed,reason in (
                (offer['anchor_reason'] != 'provisional contour','ambiguous foreground level'),
                (offer['gain'] >= .08,'insufficient merit gain'),
                (best['coverage'] >= .65,'weak supported cell coverage'),
                (best['spill'] <= .04,'unassigned-pixel spill'),
                (best['spill'] <= prior['spill']+.01,'increased spill')) if not passed]
            assert offer['eligible'] == (not failed)
            offer['gate_failures'] = failed
        reference_levels(frame,plan)
        base=dict(geometry=frame['baseline']['geometry'],scores=frame['baseline']['scores'],
                  pack_sha256=frame['pack'].get('member_sha256',frame['pack']['sha256']),paired=ZERO)
        record=dict(id=key,dataset=frame['dataset'],split=frame['split'],rgb=frame['rgb'],
                    reference=frame['reference'],camera_usable=frame['baseline']['camera_usable'],
                    base=base,plan=plan,choice=dict.fromkeys(('base','source','chooser'),'base'))
        assert record['camera_usable'] and plan['pack_sha256']==base['pack_sha256']
        offers=plan.get('audit',{}).get('offers',[])
        if offers:
            control=read(OUT/key/'control.json')
            assert control['decoded_sha256']==frame['decoded_sha256']
            record['verified_noop']=control
            alt=read(OUT/key/'alternate.json')
            record['alt']=alt
            if offers[0]['eligible']:
                record['choice']['source']='alt'
            if alt['geometry']['strict']['f1']>base['geometry']['strict']['f1']+1e-12:
                record['choice']['chooser']='alt'
        records.append(record)
    save(REVIEW/'cases.json.gz',records)
    groups={'all':records,'development':records[:4],'regression':records[4:]}
    groups.update({k:[r for r in records if r['dataset']==k] for k in controls['pilot']['datasets']})
    stats={g:{p:pool(rs,p) for p in ('base','source','chooser')} for g,rs in groups.items()}
    save(REVIEW/'summary.json',dict(groups=stats,controls_sha256=digest(CONTROL),cases_sha256=digest(REVIEW/'cases.json.gz'),
                                    cohort_available={k:v['available'] for k,v in controls['pilot']['datasets'].items()}))
    overlay(records)


if __name__=='__main__':
    main()
