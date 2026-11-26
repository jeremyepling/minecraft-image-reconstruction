"""Verify the frozen release packs and summarize paired visible-geometry changes."""
import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import ROOT, checked, digest, read, save

SIZES = {'validation':12, 'test':12, 'regression':56, 'native':12, 'old':12, 'fresh':8, 'train':72}


def summarize(rows):
    counts = Counter()
    directions = Counter()
    for row in rows:
        m = row['relative_geometry']
        counts.update(m)
        before = m['correct']-m['gained']+m['lost']
        extras = m['extras']-m['extra_added']+m['extra_removed']
        cross = m['correct']*(before+extras+m['target'])-before*(m['placed']+m['target'])
        directions['improved' if cross>0 else 'regressed' if cross<0 else 'tied'] += 1
    correct = counts['correct']
    previous = correct-counts['gained']+counts['lost']
    previous_extra = counts['extras']-counts['extra_added']+counts['extra_removed']
    return dict(counts,frames=len(rows),**directions,precision=correct/counts['placed'],recall=correct/counts['target'],
                f1=2*correct/(counts['placed']+counts['target']),
                previous_f1=2*previous/(previous+previous_extra+counts['target']))


def run(root,out):
    out.mkdir(exist_ok=True,parents=True)
    all_rows = {}
    reference = ROOT/'benchmark/reviews/relative-build-01/scores.zip'
    with ZipFile(reference) as z:
        prior = {(r['split'],r['id']):r for r in json.loads(z.read('scores.json'))['cases']}
    with ZipFile(out/'scores.zip','w',ZIP_DEFLATED) as z:
        for split,n in SIZES.items():
            source = root/split
            cases = read(source/'cases.json')
            assert len(cases)==len({r['id'] for r in cases})==n
            for row in cases:
                key = row['id']
                old = prior[split,key]
                assert old['rgb_sha256']==row['rgb_sha256']
                for name in ('input_sha256','surface_sha256','counts_sha256'):
                    if name in old:
                        assert old[name]==row[name]
                m = row['relative_geometry']
                prior_counts = old['methods']['current']['aligned']
                assert prior_counts['correct']==m['correct']-m['gained']+m['lost']
                assert prior_counts['extras']==m['extras']-m['extra_added']+m['extra_removed']
                pack = checked(source/key/'reconstruction.zip',row['pack_sha256'])
                local = set(decoded_pack(pack))
                cells = source/key/'cells.json'
                assert local==set(map(tuple,read(cells)['local']))
                z.write(pack,f'{split}/{key}/reconstruction.zip')
                z.write(cells,f'{split}/{key}/cells.json')
            for name in ('cases.json','manifest.json'):
                z.write(source/name,f'{split}/{name}')
            all_rows[split] = cases
    result = dict(primary='visible block F1 after one integer translation per build',reference_sha256=digest(reference),
                  archive_sha256=digest(out/'scores.zip'),cohorts={k:summarize(v) for k,v in all_rows.items()},
                  evaluation=summarize([r for k,rows in all_rows.items() if k!='train' for r in rows]))
    save(out/'summary.json',result)
    print(result['evaluation'])


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    run(args.predictions,args.out)

