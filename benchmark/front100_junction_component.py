"""Compare ordinary multi-sprite components using the same groups, borders and decoded scorer."""
import argparse
from pathlib import Path
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_v1_expanded as expanded
from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_scores import load_reference
from benchmark.floating_expanded_run import score_export
from benchmark.floating_selective_replay import assemble_arm
from benchmark.front100_junction_replay import portable, report
from benchmark.mask_blocks import decoded_pack
from benchmark.floating_package_rechecks import native_mask
from engine.geometry import project, UNIT
from runtime.mode import image_only


def run_option(packet, planes, rgb, destination, target):
    if (destination/'assembly.json').is_file():
        return read(destination/'assembly.json')
    original = expanded.with_visible_export
    def choose_after_border(solver):
        joined = original(solver)
        def choose(charts, **settings):
            result = joined(charts, **settings)
            groups = result['groups']
            if target is not None:
                index = next(i for i, g in enumerate(groups) if target in g['origins'])
                groups.insert(0, groups.pop(index))
            return result
        return choose
    with image_only(), patch.object(expanded, 'with_visible_export', choose_after_border):
        return assemble_arm(packet, planes, rgb, {}, 2.5, destination, save_trace=True)


def projected_coverage(charts, selected, planes, rgb):
    """Count distinct observed source samples supporting the actual exported chart faces."""
    footprint=np.zeros(((rgb.shape[0]+3)//4,(rgb.shape[1]+3)//4),dtype='uint8')
    by_id={p['id']:p for p in planes}
    for chart in charts:
        if chart['id'] not in selected:
            continue
        surface=chart['id'].split('~p')[0]
        if surface not in by_id:
            continue
        support=native_mask(by_id[surface],rgb.shape[1::-1])[::4,::4]
        uv=chart.get('export_uv',[(c['u'],c['v']) for c in chart['cells']])
        if not uv:
            continue
        quads=project(np.asarray(chart['H']),np.asarray(uv)[:,None,:]+UNIT)
        layer=np.zeros_like(footprint)
        for quad in quads:
            if np.isfinite(quad).all():
                cv2.fillConvexPoly(layer,np.rint((quad-.5)/4).astype('int32'),1)
        footprint |= layer & support
    return int(footprint.sum())


def measure(entry, original, multi, out):
    src = multi/'original'/entry['id']
    base = original/'original'/entry['id']
    dest = out/entry['id']
    dest.mkdir(parents=True,exist_ok=True)
    row=read(src/'case.json')
    before=read(base/'assembly.json')
    after=read(src/'assembly.json')
    changed=set(row['changed'])
    replaced=(changed & set(before['selected'])) - set(after['selected'])
    if not replaced or (changed & set(after['selected'])):
        save(dest/'options.json', dict(options=[],reason='no independently corrected original component was displaced'))
        return
    packet=read(src/'charts.json')
    planes=read(base/'source-handoff/surfaces.json')['planes']
    image_path=Path(entry['variants']['original']['image'])
    if digest(image_path)!=row['source_sha256'] or digest(src/'charts.json')!=row['charts_sha256']:
        raise ValueError('Frozen handoff mismatch')
    rgb=np.asarray(Image.open(image_path).convert('RGB'))
    ordinary=run_option(packet,planes,rgb,dest/'ordinary',None)
    if read(dest/'ordinary/assembly.json')!=after or decoded_pack(dest/'ordinary/reconstruction.zip')!=decoded_pack(src/'reconstruction.zip'):
        raise ValueError('Component no-op changed the ordinary export: '+entry['id'])
    options=[]
    reference=load_reference(entry)
    candidates = [('ordinary',dest/'ordinary',ordinary,row['score'])]
    for chart_id in sorted(replaced):
        choice=dest/chart_id
        record=run_option(packet, planes,rgb,choice,chart_id)
        score, aligned=score_export(entry,choice/'reconstruction.zip',record,reference=reference)
        save(choice/'aligned-placements.json.gz',[[*cell,b] for cell,b in sorted(aligned.items())])
        candidates.append((chart_id,choice,record,score))
    for name,folder,record,score in candidates:
        oriented=read(folder/'oriented-charts.json')['charts']
        pixels=projected_coverage(oriented,record['selected'],planes,rgb)
        options.append(dict(name=name,selected=record['selected'],source_samples=pixels,score=score))
    save(dest/'options.json',dict(options=options,changed=sorted(changed),displaced=sorted(replaced),
                                  oracle=max(options,key=lambda c:c['score']['geometry']['strict']['f1'])['name']))
    print(entry['id'],[(o['name'],o['score']['geometry']['strict'],o['score']['scores']['strict']['f1']) for o in options],flush=True)


def select_by_coverage(packet, planes, rgb, before, after, certified, work):
    """Keep the corrected original group if it covers more visible pixels than its unsupported replacement."""
    lost=(certified & set(before['selected']))-set(after['selected'])
    if not lost or certified & set(after['selected']):
        return None,dict(displaced=sorted(lost),offers=[])
    ordinary=run_option(packet,planes,rgb,work/'ordinary',None)
    if read(work/'ordinary/assembly.json')!=after:
        raise ValueError('Current component differs from ordinary replay')
    def source_support(name,assembly,path):
        oriented=read(path/'oriented-charts.json')['charts']
        return dict(name=name,visible_samples=projected_coverage(oriented,assembly['selected'],planes,rgb),
                    selected=assembly['selected'])
    ordinary_evidence=source_support('ordinary',ordinary,work/'ordinary')
    offers=[ordinary_evidence]
    for target in sorted(lost):
        folder=work/target
        assembly=run_option(packet,planes,rgb,folder,target)
        offers.append(source_support(target,assembly,folder))
    # Ties keep the ordinary group. Both scales were fully measured from the source.
    winner=max(offers,key=lambda item:item['visible_samples'])
    return (None if winner is ordinary_evidence else work/winner['name']),dict(displaced=sorted(lost),offers=offers)


def source_case(job):
    entry,base,multi,out=job
    target=out/'original'/entry['id']
    target.mkdir(parents=True,exist_ok=True)
    parent=multi/'original'/entry['id']
    default=base/'original'/entry['id']
    previous=read(parent/'case.json')
    src=Path(entry['variants']['original']['image'])
    packet=read(parent/'charts.json')
    if digest(src)!=entry['rgb_sha256'] or digest(parent/'charts.json')!=previous['charts_sha256']:
        raise ValueError('Source/chart digest mismatch')
    after=read(parent/'assembly.json')
    before=read(default/'assembly.json')
    certified=set(previous['changed'])
    # Each changed chart on this fixed arm must carry three leave-one-out verified full sprites.
    guard=read(parent/'evidence.json')
    if any(len(guard[c]['selected']['independent_tiles'])<3 for c in certified):
        raise ValueError('Displaced chart lacks full-sprite evidence')
    rgb=np.asarray(Image.open(src).convert('RGB'))
    planes=read(parent/'source-handoff/surfaces.json')['planes']
    with image_only():
        chosen,evidence=select_by_coverage(packet,planes,rgb,before,after,certified,target)
    evidence['choice']=chosen.name if chosen else 'ordinary'
    save(target/'evidence.json',evidence)
    folder=chosen or parent
    for name in ('assembly.json','reconstruction.zip','oriented-charts.json'):
        shutil.copyfile(folder/name,target/name)
    shutil.copyfile(parent/'charts.json',target/'charts.json')
    score=previous['score']
    stages=dict(previous['stages'])
    paired=Counter(previous['paired'])
    if chosen:
        record=read(target/'assembly.json')
        reference=load_reference(entry)
        score,aligned=score_export(entry,target/'reconstruction.zip',record,reference=reference)
        save(target/'aligned-placements.json.gz',[[*cell,b] for cell,b in sorted(aligned.items())])
        old={tuple(line[:3]) for line in read(default/'aligned-placements.json.gz')}
        fresh=set(aligned)
        truth=set(reference.truth)
        paired=dict(correct_gained=len((fresh-old)&truth),correct_lost=len((old-fresh)&truth),
                    extras_added=len((fresh-old)-truth),extras_removed=len((old-fresh)-truth))
        stages.update(selected_charts=len(record['selected']),recorded_joins=len(record['joins']))
    else:
        shutil.copyfile(parent/'aligned-placements.json.gz',target/'aligned-placements.json.gz')
    result=dict(previous,score=score,stages=stages,paired=paired,selected_component=evidence['choice'],
                assembly_sha256=digest(target/'assembly.json'),oriented_charts_sha256=digest(target/'oriented-charts.json'))
    save(target/'case.json',result)
    portable((entry,base,out))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','multispot','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    entries=read(args.baseline/'fixtures.json')['cases']
    if args.source:
        args.out.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.baseline/'fixtures.json',args.out/'fixtures.json')
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(source_case,[(entry,args.baseline,args.multispot,args.out) for entry in entries]))
        report(entries,args.baseline,args.out)
    else:
        for entry in entries:
            measure(entry,args.baseline,args.multispot,args.out)


if __name__=='__main__':
    main()
