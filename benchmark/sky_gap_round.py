"""Compare a row-calibrated sky veto against the current finder on the frozen 56 screenshots."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from benchmark.afterpass_round import COHORTS, picture_table
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_sheet import label
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL
from benchmark.normal_round import groups
from engine.sky_gap import METHOD, SETTINGS, gap_mask
from benchmark.sky_round import occupied, reference_masks
from benchmark.split_round import save_and_score
from benchmark.surface_reference import ARCHIVE
from benchmark.wall_join_test import checked, read_gzip
from engine.sky_filter import filter_surface, sky_masks
from runtime.mode import image_only, mark_image_only

BASE = ROOT / 'benchmark/reviews/sky-01'
REVIEW = ROOT / 'benchmark/reviews/sky-ambiguity-01/index.json'
NAMES = dict(current='Current sky exclusion', gap='Same-row sky in openings')


def evaluate(job):
    case, row, sha, selection, out = job
    with Image.open(checked(case.path('image'), row['sha256'])) as src:
        photo = src.convert('RGB')
    archive = read_gzip(checked(BASE/row['predictions'], sha))
    parents = archive['sky']
    start = perf_counter()
    blue, cloud = sky_masks(photo)
    gap = gap_mask(photo, blue, cloud)
    if gap.dtype != np.bool_ or gap.shape != blue.shape or np.any(gap & (blue | cloud)):
        raise ValueError('Expected a separate native-resolution mask of newly identified sky')
    parts, actions = filter_surface(photo, parents, masks=(gap, np.zeros_like(gap)))
    elapsed = perf_counter()-start
    predictions = dict(current=parents, gap=parts)
    result = save_and_score(case, row, predictions, dict(current=0., gap=elapsed), out, 'sky')
    score_dimensions(case, row, predictions, result, baseline_method='sky')
    before, after = [occupied(proposals, photo.size) for proposals in predictions.values()]
    if not np.array_equal(after, before & ~gap):
        raise ValueError('Candidate failed the native-pixel subtraction check')
    building, sky, _ = reference_masks(case)
    removed = before & ~after
    result['native_pixels'] = dict(sky_before=int((sky & before).sum()), sky_after=int((sky & after).sum()),
                                    removed_building=int((removed & building).sum()), removed_total=int(removed.sum()),
                                    building_before=int((before & building).sum()), flagged_building=int((gap & building).sum()))
    result['filter_actions'] = actions
    if selection:
        if selection['sha256'] != row['sha256']:
            raise ValueError('Changed review frame')
        l,t,r,b = selection['box']
        if selection['id'] in 'ABCD':
            marked = before & sky
            left = after & marked
            state = 'still passed as surface'
        else:
            marked = occupied(archive['current'], photo.size) & ~before & building
            left = marked & ~after
            state = 'still excluded'
        def local(mask):
            return int(mask[t:b,l:r].sum())
        if local(marked) != selection['marked'] or int(marked.sum()) != selection['in_view']:
            raise ValueError('Could not reproduce the reviewed pixels')
        result['review'] = dict(**selection, current_marked=local(marked), candidate_remaining=local(left),
                                current_full=int(marked.sum()), candidate_full=int(left.sum()), state=state)
    return result


def render(out, rows, cases, selected):
    cards = {r['review']['id']:r for r in rows if 'review' in r}
    page = Image.new('RGB',(1130,2250),'#f7f5ef')
    pen = ImageDraw.Draw(page)
    def text(x,y,s,size,bold=False):
        label(pen,(x,y),s,size,fill='#173139',bold=bold)
    text(25,20,'Can the finder exclude sky inside openings?',38,True)
    text(25,82,'Original review windows, viewed at identical coordinates and zoom.',23)
    text(25,120,'Orange = sky still claimed as surface. Pink = previously removed disputed roof pixel.',22)
    for index, view in enumerate(selected):
        tag=view['id']
        row=cards[tag]
        key=row['key']
        photo=Image.open(cases[key].path('image')).convert('RGB')
        building,sky,_=reference_masks(cases[key])
        prediction=read_gzip(out/row['predictions'])
        before,after=[occupied(prediction[m],photo.size) for m in NAMES]
        if tag in 'EF':
            prior=read_gzip(BASE/row['predictions'])
            old=occupied(prior['current'],photo.size)
            before,after=[old & building & ~m for m in (before,after)]
            color=[255,42,136]
        else:
            before,after=before & sky,after & sky
            color=[255,156,0]
        box=tuple(view['box'])
        l,t,r,b=box
        clip=photo.crop(box)
        y=185+index*344
        text(25,y,tag+'   '+key.split('/',1)[1].split('-',1)[1].replace('-',' ').title(),26,True)
        for j,(name,mask) in enumerate((('Original pixels',None),('Current filter',before),('Same-row filter',after))):
            x=25+369*j
            text(x,y+46,name,21,True)
            rgb=np.array(clip)
            if mask is not None:
                rgb[mask[t:b,l:r]]=color
            page.paste(Image.fromarray(rgb).resize((350,round(350*clip.height/clip.width)),Image.Resampling.NEAREST),(x,y+78))
            if mask is not None:
                n=int(mask[t:b,l:r].sum())
                text(x,y+282,f'{n} pixels '+row['review']['state'],18)
    page.save(out/'comparison.png',optimize=True)


@image_only()
def run(out,workers):
    source=read(BASE/'manifest.json')
    rows=read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    checked(ARCHIVE,source['reference_sha256'])
    index=read(REVIEW)
    if source['protocol']!=PROTOCOL or digest(BASE/'manifest.json')!=index['source_manifest']:
        raise ValueError('Prior review or scoring changed')
    groups_by_split,suite,fixtures=groups()
    cases={c.floor+'/'+c.id:c for grp in groups_by_split.values() for c in grp}
    frames={r['key']:r['sha256'] for r in rows}
    if len(rows)!=56 or len(set(frames.values()))!=52 or frames.keys()!=cases.keys() or fixtures!=source['fixtures'] or frames!={r['key']:r['sha256'] for r in source['frames']}:
        raise ValueError('Different frozen suite')
    selected={v['frame']:v for v in index['examples']}
    out.mkdir(parents=True,exist_ok=False)
    code=revision()
    jobs=[(cases[r['key']],r,source['artifacts'][r['predictions']],selected.get(r['key']),out) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results=list(pool.map(evaluate,jobs))
    table={}
    for grp in ('all',*COHORTS,*sorted({r['layout'] for r in rows})):
        subset=[r for r in results if grp in ('all',r['cohort'],r['layout'])]
        counts=Counter()
        for row in subset:
            counts.update(row['native_pixels'])
        table[grp]=dict(**summary(subset,NAMES),native_pixels=dict(counts))
    save(out/'cases.json',results)
    save(out/'scores.json',table)
    (out/'pictures.md').write_text(picture_table(results,NAMES,NAMES).replace('same default','same sky-filtered masks'),encoding='utf8')
    render(out,results,cases,index['examples'])
    save(out/'manifest.json',dict(run_id=out.name,code=code,version=suite['version'],frames=source['frames'],
        fixtures=fixtures,reference_sha256=source['reference_sha256'],protocol=PROTOCOL,splits=source['splits'],
        capture=source['capture'],methods=NAMES,baseline='clear-noon-sky-v1',selection='Frozen before replay; reviewed A-D are regression examples.',
        source_manifest_sha256=digest(BASE/'manifest.json'),source_manifest=str(BASE.relative_to(ROOT).as_posix())+'/manifest.json',
        review_index_sha256=digest(REVIEW),candidate=METHOD,settings=SETTINGS,candidate_module='engine.sky_gap.gap_mask',
        candidate_sha256=digest(ROOT/'engine/sky_gap.py'),source_model_timing=source['source_model_timing'],new_inference_s=0,
        timing='Cached time includes exterior sky detection, gap detection and incremental subtraction. I/O and scoring excluded.',
        reference='Only after prediction: upward rays outside the recorded building are sky on this flat world.',
        reproduce='python -m benchmark.sky_gap_round --out runs/sky-gap-replay',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(table['all']['native_pixels'],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,default=ROOT/'runs'/f'sky-gap-{uuid4().hex[:12]}')
    cli.add_argument('--workers',type=int,default=3)
    args=cli.parse_args()
    run(args.out.resolve(),args.workers)
