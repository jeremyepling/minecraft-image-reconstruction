"""Package the same-input mask, count, depth and chooser comparisons."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark.inset_decision_model import SPLITS
from benchmark.texture_camera_eval import checked, digest, read, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/facade-inset-round-01'))
    args=parser.parse_args()
    archive=args.out/'candidate-scores.zip'
    report={}
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
        chooser=args.runs/'chooser/rows.json'
        assert digest(chooser)==read(args.out/'model.json')['source_sha256']
        z.write(chooser,'chooser/rows.json')
        for method in ('returns','phase','phase-guarded','tree'):
            run=args.runs/method
            meta=read(run/'summary.json')
            assert meta['frames']==204 and meta['baseline']=='c589f1a'
            source=checked(run/'cases.json',meta['cases_sha256'])
            rows=read(source)
            z.write(source,method+'/cases.json')
            z.write(run/'summary.json',method+'/summary.json')
            groups={}
            for cohort,splits in SPLITS.items():
                totals=Counter()
                for row in rows:
                    if row['split'] not in splits:
                        continue
                    totals['frames']+=1
                    if not row['changed']:
                        continue
                    before,after,geometry=row['before'],row['after'],row['geometry']
                    assert geometry['baseline_reproduced']
                    totals['changed']+=1
                    totals['surfaces_gained']+=len(set(after['matched'])-set(before['matched']))
                    totals['surfaces_lost']+=len(set(before['matched'])-set(after['matched']))
                    totals['unmatched_delta']+=after['unmatched_predictions']-before['unmatched_predictions']
                    totals['correctly_paired_pixel_delta']+=after['paired_pixels']-before['paired_pixels']
                    totals['background_delta']+=after['background_pixels']-before['background_pixels']
                    for k in ('gained','lost','extra_added','extra_removed','hidden_added'):
                        totals[k]+=geometry[k]
                groups[cohort]=dict(totals)
            report[method]=groups
    save(args.out/'challengers.json',dict(baseline='c589f1a',archive_sha256=digest(archive),methods=report))


if __name__=='__main__':
    main()
