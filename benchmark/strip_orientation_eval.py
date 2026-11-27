"""Compare photographed strip interpretations against the published edb10a3 geometry."""
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from benchmark import wall_section_eval as wall
from benchmark import framed_depth_eval as framed, roof_span_eval as span
from benchmark.facade_visibility import rates
from benchmark.roof_span_worst_stages import read_view
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import ROOT,checked,read,save

REVIEW=ROOT/'benchmark/reviews/visible-error-budget-01'
ARCHIVE=REVIEW/'comparisons.zip'
CONTROL='edb10a3'
SLIVERS=ROOT/'benchmark/reviews/strip-orientation-01/sliver-counts.json'


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


@lru_cache(maxsize=1)
def fixed_slivers():
    return {r['id']:r for r in read(SLIVERS)['frames'] if r['split'] in framed.EVALUATION}


def add_priority(job,row):
    split,frame,prior,out,_=job
    if split not in framed.EVALUATION:
        return row
    baseline=fixed_slivers()[frame['id']]
    assert baseline['pack_sha256']==prior['pack_sha256'] and baseline['rgb_sha256']==row['rgb_sha256']
    if row['pack_sha256']==baseline['pack_sha256']:
        hits,misses,*_=baseline['stats']['face_4_horizontal']
    else:
        _,_,_,_,_,labels,ids,truth,_,placed,_=read_view(out,split,frame,prior,row)
        neutral=horizontal_slivers(labels,ids)
        hits=len(neutral&placed)
        misses=len(neutral-placed)
        assert neutral<=truth
    m=row['relative_geometry']
    row['facade_priority']=dict(correct=m['correct']-hits,missed=m['missed']-misses,
                                extras=m['extras'],optional_correct=hits,optional_missed=misses)
    return row


def score(job):
    split,frame,prior,_,predictions=job
    folder='production/'+split
    if f"{folder}/{frame['id']}/reconstruction.zip" not in archive().NameToInfo:
        return add_priority(job,wall.score(job))
    if framed.visible_depth_score.predict is None or predictions is not None:
        framed.photo(split,frame,prior)
    return add_priority(job,framed.score_archive(job,archive(),folder))


def report_priority(out):
    meta=read(out/'summary.json')
    rows=[r['facade_priority'] for r in read(checked(out/'cases.json',meta['cases_sha256'])) if 'facade_priority' in r]
    if rows:
        m={k:sum(r[k] for r in rows) for k in rows[0]}
        score_f1=rates(m['correct'],m['missed'],m['extras'])['f1']
        meta['facade_priority']=dict(views=len(rows),**m,f1=score_f1,
            scope='Fixed structure views in this run; individual horizontal-only faces without a native 4x4 patch are neutral')
        save(out/'summary.json',meta)
        print(f"facade-priority {len(rows)} {score_f1:.4%}",flush=True)


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
        report_priority(args.out)
