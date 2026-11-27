"""Evaluate each committed crop edit against fixed frontmost reference masks after inference."""
import argparse
from pathlib import Path

import numpy as np

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.sky_crop_chooser import DEV
from benchmark.sky_crop_options import ACTIVE, native


OUT = ROOT/'runs/sky-crop-correction-policy'
REF = WORKTREES/'sky-crop-reference/runs/sky-crop-reference-01'
ARCHIVE = ROOT/'benchmark/reviews/sky-crop-correction-policy-01'


def summarize(rows):
    fields = ('tp','fp','fn','add_correct','add_wrong','remove_correct','remove_wrong')
    all_steps = [[r['steps'][i] for r in rows] for i in range(5)]
    return [dict(crops=len(step),mean_iou=float(np.mean([s['iou'] for s in step])),
                 **{k:sum(r[k] for r in step) for k in fields},
                 gained=sum(s['iou_delta']>0 for s in step),lost=sum(s['iou_delta']<0 for s in step),
                 tied=sum(s['iou_delta']==0 for s in step)) for step in all_steps]


def score(folder):
    baseline = read(REF/'report.json')
    truth = read(REF/'exact-target-masks.json')['frames']
    packets = [read(folder/f'round-{i}.json')['frames'] for i in range(5)]
    traced = read(folder/'trace.json')
    scoped = read(folder/'selected.json')['frames']
    rows, excluded, hashes = [], [], {}
    for case in baseline['cases']:
        case_id = case['id']
        ordinary_path = ACTIVE/case_id/'source-handoff/surfaces.json'
        hashes[case_id] = dict(source_sha256=case['source_sha256'],reference_sha256=case['reference_sha256'],
                               surfaces_sha256=digest(ordinary_path))
        if hashes[case_id]['surfaces_sha256'] != case['surfaces_sha256']:
            raise ValueError(f'Mismatched frozen handoff on {case_id}')
        parts = {p['id']:p['support'] for p in read(ordinary_path)['planes']}
        for p in case['arms']['baseline']['proposals']:
            pid = p['id']
            if pid not in truth[case_id]:
                if p['target'] is not None:
                    raise ValueError(f'Missing paired reference {pid}')
                scoped[case_id].pop(pid,None)
                excluded.append([case_id,pid])
                continue
            target = truth[case_id][pid]
            ref = native(target,tuple(target['size']))
            states = []
            for step,packet in enumerate(packets):
                chosen = packet[case_id].get(pid,parts[pid])
                current = native(chosen,tuple(target['size']))
                tp = int((ref & current).sum())
                fp = int((~ref & current).sum())
                fn = int((ref & ~current).sum())
                iou = tp/(tp+fp+fn)
                previous = before if step else current
                added = current & ~previous
                removed = previous & ~current
                delta = iou-states[-1]['iou'] if step else 0.
                states.append(dict(step=step,iou=iou,iou_delta=delta,tp=tp,fp=fp,fn=fn,
                                   add_correct=int((added&ref).sum()),add_wrong=int((added&~ref).sum()),
                                   remove_correct=int((removed&~ref).sum()),remove_wrong=int((removed&ref).sum())))
                if step:
                    trace = traced['frames'][case_id][pid][step-1]
                    if (int(added.sum()),int(removed.sum()))!=(trace['added'],trace['removed']):
                        raise ValueError('Replay edits differ from recorded source-only updates')
                before = current
            rows.append(dict(case=case_id,id=pid,label=p['label'],target=p['target'],steps=states))
    if len(rows)!=188 or len(excluded)!=5:
        raise ValueError('Unexpected fixed-target inventory')
    summary = dict(case={c:summarize([r for r in rows if r['case']==c]) for c in truth},
                   development=summarize([r for r in rows if r['case'] in DEV]),
                   held_out=summarize([r for r in rows if r['case'] not in DEV]),all=summarize(rows))
    save(folder/'selected-paired.json',dict(frames=scoped))
    result = dict(initialization=traced['initialization'],step_selected_on_development=traced['selected_round'],
                  model_sha256=traced['model_sha256'],inputs=hashes,excluded_no_reference=excluded,
                  matched_proposals=188,raw_proposals=193,summary=summary,trajectories=rows)
    save(folder/'paired-scores.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    for name,folder in [('ordinary',args.output),('chooser',args.output/'from-chooser')]:
        receipt = score(folder)
        save(ARCHIVE/f'{name}-scores.json.gz',receipt)
        save(ARCHIVE/f'{name}-trajectory.json.gz',dict(trace=read(folder/'trace.json'),
             rounds=[read(folder/f'round-{i}.json') for i in range(5)]))
        save(ARCHIVE/f'{name}-selected.json.gz',read(folder/'selected.json'))
        save(ARCHIVE/f'{name}-selected-paired.json.gz',read(folder/'selected-paired.json'))
        print(name,[(s,[round(r['mean_iou'],5) for r in receipt['summary'][s]])
                    for s in ('development','held_out','all')],flush=True)
    save(ARCHIVE/'freeze.json',read(args.output/'freeze.json'))
    (ARCHIVE/'forest.xml.gz').write_bytes((args.output/'forest.xml.gz').read_bytes())


if __name__=='__main__':
    main()
