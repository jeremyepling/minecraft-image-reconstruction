"""Compare photographed strip interpretations against the published edb10a3 geometry."""
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from benchmark import wall_section_eval as wall
from benchmark import framed_depth_eval as framed, roof_span_eval as span
from benchmark.texture_camera_eval import ROOT,checked,read

REVIEW=ROOT/'benchmark/reviews/visible-error-budget-01'
ARCHIVE=REVIEW/'comparisons.zip'
CONTROL='edb10a3'


@lru_cache(maxsize=1)
def archive():
    return ZipFile(checked(ARCHIVE,read(REVIEW/'results.json')['archive_sha256']))


def cases(splits):
    rows=wall.height.previous_rows('production',archive())
    index={(r['split'],r['id']):r for r in rows}
    assert len(index)==len(rows)==sum(framed.SIZES.values())
    for split,frame,previous in wall.cases(splits):
        r=index[split,frame['id']]
        assert r['control_pack_sha256']==previous['pack_sha256'] and r['rgb_sha256']==previous['rgb_sha256']
        assert r['reference_hashes']==frame['files'] and r['input_hashes']==previous['input_hashes']
        yield split,frame,dict(r,**r['input_hashes'])


def score(job):
    split,frame,prior,_,predictions=job
    folder='production/'+split
    if f"{folder}/{frame['id']}/reconstruction.zip" not in archive().NameToInfo:
        return wall.score(job)
    if framed.visible_depth_score.predict is None or predictions is not None:
        framed.photo(split,frame,prior)
    return framed.score_archive(job,archive(),folder)


if __name__=='__main__':
    parser=framed.parser(__doc__)
    parser.add_argument('--transfer-inputs',type=Path)
    args=parser.parse_args()
    if args.transfer_inputs:
        wall.roof_height_transfer.run(args.transfer_inputs.resolve(),args.candidate_root.resolve(),args.candidate,
            args.out,bundle=archive(),prefix='production/transfer',control=CONTROL,archive_path=ARCHIVE)
    else:
        span.compare(args,case_source=cases,scorer=score,control=CONTROL,default_engine=CONTROL,
                     additional_sources=(wall.height.CHANGES,wall.facade.ARCHIVE,wall.ARCHIVE,ARCHIVE))
