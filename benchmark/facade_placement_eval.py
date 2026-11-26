"""Replay the published visible-only roof-height default for the placement-order and rectangle trials."""
import json
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from benchmark import framed_depth_eval as framed, roof_height_eval as height, roof_height_transfer, roof_span_eval as span
from benchmark.texture_camera_eval import ROOT, checked, read

REVIEW=ROOT/'benchmark/reviews/roof-height-01'
ARCHIVE=REVIEW/'comparisons.zip'
CONTROL='1c2f77d'


@lru_cache(maxsize=1)
def archive():
    return ZipFile(checked(ARCHIVE,read(REVIEW/'results.json')['archive_sha256']))


def cases(splits):
    rows=height.previous_rows('hybrid',archive())
    index={(r['split'],r['id']):r for r in rows}
    assert len(rows)==len(index)==sum(framed.SIZES.values())
    for split,frame,prior in height.cases(splits):
        row=index[split,frame['id']]
        assert row['control_pack_sha256']==prior['pack_sha256']
        assert row['rgb_sha256']==prior['rgb_sha256'] and row['reference_hashes']==frame['files']
        assert row['input_hashes']==prior['input_hashes']
        yield split,frame,{**row,**row['input_hashes'], 'previous_pack_changed':prior['changed']}


def score(job):
    split,frame,prior,_,predictions=job
    if framed.visible_depth_score.predict is None or predictions is not None:
        framed.photo(split,frame,prior)
    if prior['changed']:
        bundle,folder=archive(),f'hybrid/{split}'
    elif prior['previous_pack_changed']:
        bundle,folder=height.changes(),f'conservative/{split}'
    else:
        bundle,folder=framed.archive(span.PACKS),split
    row=framed.score_archive(job,bundle,folder)
    if 'roof' in prior:
        assert json.loads(json.dumps(row['previous_roof']))==prior['roof'],frame['id']
    return row


if __name__=='__main__':
    parser=framed.parser(__doc__)
    parser.add_argument('--transfer-inputs',type=Path,help='Replay the four shelter/gate photos instead.')
    args=parser.parse_args()
    if args.transfer_inputs:
        roof_height_transfer.run(args.transfer_inputs.resolve(),args.candidate_root.resolve(),args.candidate,args.out,
                                 bundle=archive(),prefix='hybrid/transfer',control=CONTROL,archive_path=ARCHIVE)
    else:
        span.compare(args,case_source=cases,scorer=score,control=CONTROL,default_engine='f295b0f',
                     additional_sources=(height.CHANGES,ARCHIVE))
