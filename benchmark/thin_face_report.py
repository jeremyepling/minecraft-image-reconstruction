"""Audit native-ray strip additions and package their decoded geometry by cohort."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

from benchmark import canopy_fixture_metrics as gap
from benchmark import relative_canopy_spacing as roof
from benchmark import visible_depth_transfer as transfer
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack,registered
from benchmark.thin_face_atlas_round import DATA, ROOT, PREVIOUS, EXTRA, checked, digest, fixture, frames, native, read, save
from training.visible_labels import visible_labels


def analyze(path,mode,out,z):
    rows=read(path/'scores.json')['rows']
    checked_cases=[]
    by_split={}
    canopy=[]
    for split in sorted({r['split'] for r in rows}):
        src=fixture(split)
        samples=(read(src/'screen.json')['selected'] if src else native.rows() if split=='native' else frames(DATA,split))
        specs=read(roof.PROBES)
        by_id={f['id']:f for f in samples}
        base=EXTRA/'candidate' if split=='extra' else PREVIOUS/split
        count=Counter()
        for r in [r for r in rows if r['split']==split]:
            key=r['id']
            f=by_id[key]
            sha=f['files']
            candidate=r['methods'][mode]
            m=candidate['metrics']
            prior_file=checked(base/key/'reconstruction.zip',r['baseline_pack_sha256'])
            pred_file=checked(path/split/key/(mode+'.zip'),candidate['sha256'])
            prior=set(decoded_pack(prior_file))
            after=set(decoded_pack(pred_file))
            assert prior<=after
            added=after-prior
            row=dict(id=key,split=split,layout=r['layout'],rgb_sha256=r['rgb_sha256'],source_sha256=r['input_sha256'],
                     baseline_pack_sha256=r['baseline_pack_sha256'],pack_sha256=candidate['sha256'],
                     proposed=r['proposed'],possible=r['possible'],metrics=m,translation=candidate['translation'])
            if added or src:
                if src or split=='native':
                    rgb=transfer.rgb_path(f) if src else native.rgb_path(f)
                    pose=read(checked(rgb.with_suffix('.json'),sha['camera.json']))
                    raw=read(checked(rgb.with_name(rgb.stem+'-blocks.json'),sha['world-readback.json']))
                    labels,ids,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in raw},(0,0,0))
                else:
                    p=DATA/'data'/split/f['directory']
                    pose=read(checked(p/'camera.json',sha['camera.json']))
                    labels=read(checked(p/'labels.json',sha['labels.json']))
                    with np.load(checked(p/'labels.npz',sha['labels.npz'])) as ref:
                        ids=ref['face_id']
                if src:
                    camera=f['camera']
                else:
                    from benchmark.texture_camera_eval import STABLE
                    camera=read(checked(STABLE/'runs/visible-depth-rounds/inputs'/(key+'.json'),r['input_sha256']))['camera']
                T=registration(np.asarray(camera['P']),pose)
                truth={tuple(b['xyz_world']) for b in labels['blocks']}
                sampled=set(np.unique(ids[2::4,2::4]))
                sampled_cells={tuple(face['xyz_world']) for face in labels['faces'] if face['id'] in sampled}
                horiz={tuple(face['xyz_world']) for face in labels['faces'] if face['direction'] in ('up','down')}
                world=registered(added,T)
                correct=world&truth
                assert not any(candidate['translation']) or not added
                nearless={p for p in added if sum((p[0]+dx,p[1],p[2]+dz) in prior
                              for dx,dz in ((-1,0),(1,0),(0,-1),(0,1)))<2}
                row['addition_breakdown']=dict(correct=len(correct),coarse_missed=len(correct-sampled_cells),
                    visible_horizontal=len(correct&horiz),zero_or_one_neighbor=len(nearless),
                    correct_zero_or_one_neighbor=len(registered(nearless,T)&truth))
                matching=[k for k in specs if k==key or (split=='extra' and k.rsplit('-',1)[0]==key.rsplit('-',1)[0])]
                if src and matching:
                    spec=specs[matching[0]]
                    occupied=set(transfer.structure(raw,64))
                    ref_faces,_=gap.raster(pose,occupied)
                    target={tuple(face['xyz_world']) for face in ref_faces}
                    probe=dict(id=key,split=split)
                    for name,data in [('previous',prior),('current',after)]:
                        cells=registered(data,T)
                        probe[name]=roof.score(cells,target,spec)
                        probe[name]['gap']=gap.summarize([gap.score(f,cells,spec)])
                    canopy.append(probe)
            count.update(m)
            count.update({'proposed':r['proposed'],'possible':r['possible']})
            before=m['correct']-m['gained']+m['lost']
            extras=m['extras']-m['extra_added']+m['extra_removed']
            change=m['correct']*(before+extras+m['target'])-before*(m['placed']+m['target'])
            count['improved' if change>0 else 'regressed' if change<0 else 'tied']+=1
            z.write(pred_file,f'{mode}/{split}/{key}/reconstruction.zip')
            checked_cases.append(row)
        by_split[split]=dict(count)
    return checked_cases,by_split,canopy


def main(args):
    args.out.mkdir(exist_ok=True,parents=True)
    results=[]
    cohorts={}
    roofs=[]
    with ZipFile(args.out/'evidence.zip','w',ZIP_DEFLATED) as archive:
        for run in args.runs:
            rr,ss,probes=analyze(run,args.mode,args.out,archive)
            results.extend(rr)
            cohorts.update(ss)
            roofs.extend(probes)
    save(args.out/'scores.json',dict(method=args.mode,cohorts=cohorts,rows=results,pack_archive_sha256=digest(args.out/'evidence.zip')))
    save(args.out/'canopies.json',roofs)
    for group,splits in [('112',('regression','test','validation','old','fresh','native')),('extra',('extra',)),('train',('train',))]:
        total=sum((Counter(cohorts[s]) for s in splits if s in cohorts),Counter())
        breakdown=sum((Counter(r.get('addition_breakdown',{})) for r in results if r['split'] in splits),Counter())
        print(group,dict(total),dict(breakdown))
        for name in ('previous','current'):
            relevant=[r[name] for r in roofs if r['split'] in splits]
            if relevant:
                print(name,{k:sum(v[k] for v in relevant) for k in ('anchored_roof_cells','visible_roof_cells','matching_triples')},
                      'closed sightlines',sum(v['gap']['foreground_filled'] for v in relevant))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',default='rank85')
    main(p.parse_args())
