"""Check shipped four/two-pixel photo-edge decisions against all frozen experiment predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from benchmark.camera_round import ROOT,digest,read,revision,save
from benchmark.ledge_join_round import score_dimensions
from benchmark.normal_round import groups
from benchmark.split_round import score_regions
from benchmark.surface_reference import ARCHIVE
from benchmark.wall_join_test import checked,read_gzip
from runtime.mode import image_only,mark_image_only


APPROVED=ROOT/'benchmark/reviews/edge-detail-stable-01'
METHOD='edge_detail:native4-stable-counts'


def verify_frame(job):
    from engine.photo_edges import refine_surfaces
    case,row,archive_sha=job
    original=read_gzip(checked(APPROVED/row['predictions'],archive_sha))
    current=original['current']
    with Image.open(checked(case.path('image'),row['sha256'])) as source:
        updated,stats=refine_surfaces(source.convert('RGB'),current)
    if updated!=original[METHOD]:
        raise ValueError('An ordered proposal differs on '+row['key'])
    proposals={'current':current,METHOD:updated}
    scores,scopes=score_regions(case,proposals,row['key'])
    result=dict(key=row['key'],scores=scores,scopes=scopes)
    score_dimensions(case,row,proposals,result,'current')
    for field in ('scores','scopes','sizes','size_changes'):
        for name in proposals:
            if result[field][name]!=row[field][name]:
                raise ValueError(f"{field} differs on {row['key']} for {name}")
    changed=sum(p['support']!=q['support'] for p,q in zip(current,updated,strict=True))
    return dict(frame=row['key'],changed_surfaces=changed,eligible=scores[METHOD]['eligible'],
                matched=scores[METHOD]['matched'],sizes=result['sizes'][METHOD],
                applied_radius_px=stats['applied_radius_px'])


@image_only()
def verify(workers):
    gold=read(APPROVED/'manifest.json')
    checked(ARCHIVE,gold['reference_sha256'])
    rows=read(checked(APPROVED/'cases.json',gold['artifacts']['cases.json']))
    cohorts,suite,fixtures=groups()
    caps={c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    frames={r['key']:r['sha256'] for r in rows}
    if (fixtures!=gold['fixtures'] or suite['version']!=gold['version'] or len(rows)!=56 or
        frames.keys()!=caps.keys() or frames!={r['key']:r['sha256'] for r in gold['frames']}):
        raise ValueError('Frozen fixture changed')
    jobs=[(caps[r['key']],r,gold['artifacts'][r['predictions']]) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results=list(pool.map(verify_frame,jobs))
    return dict(code=revision(),source_run=gold['run_id'],approved_code=gold['code'],
        source_manifest_sha256=digest(APPROVED/'manifest.json'),frames=len(rows),
        scope='Archived ordinary masks after neighboring-strip cleanup; pre-crop production refinement compared with frozen final masks. Full model inference independently verified on one renamed screenshot.',
        all_ordered_proposal_records_equal=True,all_surface_and_count_scores_equal=True,
        changed_frames=sum(bool(r['changed_surfaces']) for r in results),
        changed_surfaces=sum(r['changed_surfaces'] for r in results),
        eligible=sum(r['eligible'] for r in results),matched=sum(r['matched'] for r in results),
        sizes={k:sum(r['sizes'][k] for r in results) for k in results[0]['sizes']},cases=results)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=6)
    args=cli.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,verify(args.workers))
