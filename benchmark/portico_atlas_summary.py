"""Compact per-picture diagnostics for counts and texture candidates."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.camera_round import digest, read, save
from benchmark.wall_join_test import read_gzip


def summarize(run, out):
    manifest=read(run/'manifest.json')
    rows=read(run/'cases.json')
    pictures=[]
    cobble=Counter()
    for frame in rows:
        preds=read_gzip(run/frame['predictions'])
        candidates=[]
        for target,pred in zip(frame['surfaces'],preds):
            if target['proposal']!=pred['proposal']:
                raise ValueError('Changed prediction ordering')
            e=pred['methods']['atlas']['evidence']['atlas_check']
            mats=target.get('materials',{})
            if 'minecraft:cobblestone' in mats:
                cobble['contains_cobblestone']+=1
                if len(mats)==1:
                    guess=target['guess']['atlas']
                    cobble['all_cobblestone']+=1
                    status='exact' if guess==target['actual'] else 'incorrect' if None not in guess else 'unresolved'
                    cobble[status]+=1
            candidates.append(dict(**target,atlas_evidence=e))
        pictures.append(dict(frame=frame['key'],sha256=frame['sha256'],cohort=frame['cohort'],candidates=candidates))
    save(out,dict(run_id=manifest['run_id'],code=manifest['code'],source_manifest_sha256=manifest['source_manifest_sha256'],
         reference_sha256=manifest['reference_sha256'],fixtures=manifest['fixtures'],settings=manifest['settings'],
         full_cases_sha256=digest(run/'cases.json'),pure_cobblestone_scores=cobble,pictures=pictures))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    summarize(a.run,a.out)
