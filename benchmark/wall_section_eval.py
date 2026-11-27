"""Compare wall section hypotheses to the published 028819a geometry on identical saved inputs."""
import json
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from benchmark import facade_placement_eval as facade, framed_depth_eval as framed
from benchmark import roof_height_eval as height, roof_height_transfer, roof_span_eval as span
from benchmark.texture_camera_eval import ROOT, checked, read

REVIEW = ROOT / 'benchmark/reviews/facade-placement-01'
ARCHIVE = REVIEW / 'comparisons.zip'
WINNER = 'roof_then_walls'
CONTROL = '028819a'


@lru_cache(maxsize=1)
def archive():
    return ZipFile(checked(ARCHIVE, read(REVIEW/'results.json')['archive_sha256']))


def cases(splits):
    rows = height.previous_rows(WINNER, archive())
    index = {(r['split'],r['id']): r for r in rows}
    assert len(index) == len(rows) == sum(framed.SIZES.values())
    for split,frame,prior in facade.cases(splits):
        row = index[split,frame['id']]
        assert row['control_pack_sha256'] == prior['pack_sha256']
        assert row['rgb_sha256'] == prior['rgb_sha256']
        assert row['reference_hashes'] == frame['files'] and row['input_hashes'] == prior['input_hashes']
        yield split,frame,{**row, **row['input_hashes']}


def score(job):
    split,frame,prior,_,predictions = job
    if framed.visible_depth_score.predict is None or predictions is not None:
        framed.photo(split,frame,prior)
    # Compact release archives contain only changed exports. The scorer verifies
    # the selected pack hash and local/world coordinates before it calls the adapter.
    for bundle,folder in ((archive(),f'{WINNER}/{split}'),(facade.archive(),f'hybrid/{split}'),
                          (height.changes(),f'conservative/{split}'),(framed.archive(span.PACKS),split)):
        if f"{folder}/{frame['id']}/reconstruction.zip" in bundle.NameToInfo:
            break
    row = framed.score_archive(job,bundle,folder)
    if 'roof' in prior:
        assert json.loads(json.dumps(row['previous_roof'])) == prior['roof']
    return row


if __name__=='__main__':
    parser = framed.parser(__doc__)
    parser.add_argument('--transfer-inputs',type=Path,help='Use the four shelter and gate photos.')
    args = parser.parse_args()
    if args.transfer_inputs:
        roof_height_transfer.run(args.transfer_inputs.resolve(),args.candidate_root.resolve(),args.candidate,args.out,
                                 bundle=archive(),prefix=f'{WINNER}/transfer',control=CONTROL,archive_path=ARCHIVE)
    else:
        span.compare(args,case_source=cases,scorer=score,control=CONTROL,default_engine=CONTROL,
                     additional_sources=(height.CHANGES,facade.ARCHIVE,ARCHIVE))
