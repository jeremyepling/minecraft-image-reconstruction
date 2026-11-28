"""Replay verified Front 100 Split-Border chart packets through a disjoint-patch atlas ruler."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_scores import load_reference
from benchmark.floating_cloud_replay import check_handoff
from benchmark.floating_expanded_run import score_export, summarize
from benchmark.floating_face_proposals import image_axes
from benchmark.floating_selective_replay import assemble_arm
from benchmark.front100_junction_scale import MIN_CELLS, challenge
from benchmark.mask_blocks import decoded_pack
from engine.floating_sky import sky_support
from runtime.mode import image_only


def run(job):
    entry, baseline, out, confirm_coverage = job
    cv2.setNumThreads(1)
    source_dir = baseline / 'original' / entry['id']
    target = out / 'original' / entry['id']
    if (target/'case.json').exists():
        return
    target.mkdir(parents=True,exist_ok=True)
    image_path = Path(entry['variants']['original']['image'])
    base = read(source_dir / 'case.json')
    source_sha = digest(image_path)
    packet = read(source_dir/'charts.json')
    if source_sha != entry['rgb_sha256'] or source_sha != base['source_sha256'] or base['charts_sha256'] != digest(source_dir/'charts.json'):
        raise ValueError('Changed source or chart packet: '+entry['id'])
    check_handoff(packet, source_dir/'source-handoff', source_sha)
    planes = read(source_dir/'source-handoff/surfaces.json')['planes']
    image = Image.open(image_path).convert('RGB')
    rgb = np.asarray(image)
    by_id = {row['id']:row for row in planes}
    axes = None
    sky = None
    audits = {}
    changed = []
    result = []
    with image_only():
        for chart in packet['charts']:
            plane_id = chart['id'].split('~p')[0]
            if chart['method']=='local-atlas-window' and len(chart['cells'])>=MIN_CELLS and plane_id in by_id and abs(chart['normal_camera'][1])<.67:
                if sky is None:
                    axes = image_axes(image,planes)
                    sky = sky_support(image)
                improved, audit = challenge(chart,by_id[plane_id],rgb,axes,sky,confirm_coverage=confirm_coverage)
                audits[chart['id']] = audit
                if improved is not chart:
                    changed.append(chart['id'])
                result.append(improved)
            else:
                result.append(chart)
        check = target/'noop'
        if not (check/'assembly.json').is_file():
            assemble_arm(packet, planes, rgb, {}, 2.5, check, save_trace=True)
        noop = read(check/'assembly.json')
        base_assembly = read(source_dir/'assembly.json')
        if (noop!=base_assembly or decoded_pack(check/'reconstruction.zip')!=decoded_pack(source_dir/'reconstruction.zip')):
            raise ValueError('No-op diverges: '+entry['id'])
        if changed:
            newpacket = dict(packet, charts=result)
            save(target/'charts.json',newpacket)
            assembly = assemble_arm(newpacket, planes, rgb, {}, 2.5, target/'candidate', save_trace=True)
            chosen = target/'candidate'
        else:
            chosen = check
            assembly = noop
        for name in ('reconstruction.zip','assembly.json','oriented-charts.json'):
            shutil.copyfile(chosen/name,target/name)

    paired = Counter(correct_gained=0,correct_lost=0,extras_added=0,extras_removed=0)
    stages = dict(base['stages'])
    if changed:
        reference = load_reference(entry)
        score, aligned = score_export(entry,target/'reconstruction.zip',assembly,reference=reference)
        previous = {tuple(row[:3]):row[3] for row in read(source_dir/'aligned-placements.json.gz')}
        save(target/'aligned-placements.json.gz',[[*cell,b] for cell,b in sorted(aligned.items())])
        old, fresh = set(previous), set(aligned)
        truth = set(reference.truth)
        paired.update(correct_gained=len((fresh-old)&truth),correct_lost=len((old-fresh)&truth),
                      extras_added=len((fresh-old)-truth),extras_removed=len((old-fresh)-truth))
        stages['selected_charts'] = len(assembly['selected'])
        stages['recorded_joins'] = len(assembly['joins'])
        stages['oriented_charts'] = len(read(chosen/'oriented-charts.json')['charts'])
    else:
        score = dict(base['score'], pack_sha256=digest(target/'reconstruction.zip'))
        shutil.copyfile(source_dir/'aligned-placements.json.gz',target/'aligned-placements.json.gz')
    row = dict(base,score=score,stages=stages,paired=paired,changed=changed,noop_parity=True,
               charts_sha256=digest(target/'charts.json') if changed else base['charts_sha256'],
               assembly_sha256=digest(target/'assembly.json'),oriented_charts_sha256=digest(target/'oriented-charts.json'))
    save(target/'evidence.json',audits)
    save(target/'case.json',row)
    print(entry['id'],list(audits),changed,dict(paired),flush=True)


def report(entries, baseline, out):
    records = [read(out/'original'/e['id']/'case.json') for e in entries if (out/'original'/e['id']/'case.json').is_file()]
    def group(rows):
        paired = Counter()
        for row in rows:
            paired.update(row['paired'])
        return dict(baseline=summarize([read(baseline/'original'/r['id']/'case.json') for r in rows]),
                    candidate=summarize(rows),paired=paired,changed=[r['id'] for r in rows if r['changed']])
    if records:
        save(out/'summary.json',dict(tested=len(records),noop_parity=sum(r['noop_parity'] for r in records),
              total=group(records),by_split={split:group([r for r in records if r['split']==split]) for split in sorted({r['split'] for r in records})}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--case',action='append')
    p.add_argument('--split')
    p.add_argument('--report',action='store_true')
    p.add_argument('--confirm-coverage',action='store_true')
    args = p.parse_args()
    entries = read(args.baseline/'fixtures.json')['cases']
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = args.out/'fixtures.json'
    if manifest.exists() and digest(manifest)!=digest(args.baseline/'fixtures.json'):
        raise ValueError('Fixture mismatch')
    shutil.copyfile(args.baseline/'fixtures.json',manifest)
    selected = [e for e in entries if (not args.split or e['split']==args.split) and
                (not args.case or e['id'] in args.case)]
    if not args.report:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run,[(e,args.baseline,args.out,args.confirm_coverage) for e in selected]))
    report(entries,args.baseline,args.out)


if __name__=='__main__':
    main()

