"""Replay the sky exclusion on the same 56 saved surface predictions, then score native pixels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from benchmark.afterpass_round import COHORTS, picture_table
from benchmark.camera_labels import screen_matrix
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_sheet import label
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL
from benchmark.normal_round import groups
from benchmark.split_round import save_and_score
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_pipeline_sheet import _picture
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from engine import sky_filter
from engine.surface_perimeter import _native_alpha
from runtime.mode import image_only, mark_image_only


BASE = ROOT / 'benchmark/reviews/crop-split-01'
NAMES = dict(current='Split-only zoom', sky='Split-only zoom with sky exclusion')


def occupied(parts, size):
    covered = np.zeros(size[::-1], bool)
    for part in parts:
        native = _native_alpha(part['support'], size)
        if native:
            alpha, (x0, y0, x1, y1) = native
            covered[y0:y1, x0:x1] |= np.asarray(alpha, dtype=bool)
    return covered


def reference_masks(case):
    pose, ref = recorded(case)
    building = _native_reference(pose, ref, read(case.path('labels'))['faces']) != 0
    # Sky truth is independent of the candidate's color decisions. A ray pointing
    # upward with no visible building hit cannot intersect the flat ground.
    up = np.linalg.inv(screen_matrix(pose)[:, :3])[1]
    y, x = np.ogrid[:building.shape[0], :building.shape[1]]
    elevation = (x+.5)*up[0] + (y+.5)*up[1] + up[2]
    return building, (elevation > 0) & ~building, (elevation > .035) & ~building


def evaluate(job):
    case, row, sha, out = job
    with Image.open(checked(case.path('image'), row['sha256'])) as src:
        photo = src.convert('RGB')
    parents = read_gzip(checked(BASE/row['predictions'], sha))['split']
    start = perf_counter()
    blue, cloud = sky_filter.sky_masks(photo)
    parts, info = sky_filter.filter_surface(photo, parents, masks=(blue, cloud))
    elapsed = perf_counter()-start
    props = dict(current=parents, sky=parts)
    result = save_and_score(case, row, props, dict(current=0., sky=elapsed), out, 'split')
    score_dimensions(case, row, props, result, baseline_method='split')
    old, new = [occupied(p, photo.size) for p in props.values()]
    if np.any(new & (~old | blue | cloud)):
        raise ValueError('Filtered surfaces expanded or retained identified sky pixels')
    building, sky, high_sky = reference_masks(case)
    r,g,b = np.asarray(photo,dtype=np.int16).transpose(2,0,1)
    # Separately frozen positive controls from the baseline audit (not passed to the filter).
    cloud_proxy = high_sky & (r>=140) & (g>=150) & (g>r+5) & (b>g+10) & (b<250)
    blue_proxy = high_sky & (g>r+20) & (b==255) & (b>g+30)
    removed = old & ~new
    masks = dict(control_cloud_before=old & cloud_proxy,control_cloud_after=new & cloud_proxy,
                 control_blue_before=old & blue_proxy,control_blue_after=new & blue_proxy,
                 flagged_blue=blue, flagged_cloud=cloud, removed_blue=removed & blue,
                 removed_cloud=removed & cloud, removed_total=removed,
                 removed_recorded_building=removed & building, removed_other_background=removed & ~(sky | building),
                 flagged_building_anywhere=building & (blue | cloud),
                 upward_background_before=old & sky, upward_background_after=new & sky,
                 total_building_coverage_before=old & building, total_building_coverage_after=new & building)
    result.update(native_pixels={name:int(mask.sum()) for name,mask in masks.items()}, filter_actions=info)
    return result


def tint(photo, mask, color):
    return Image.composite(Image.blend(photo, Image.new('RGB',photo.size,color),.72), photo, Image.fromarray(mask))


def render(out, rows, cases):
    # Choose the strongest actual cloud intersection in each layout, then the most sky removed.
    layouts = ('shopfront','wing','courtyard','portico','workshop','crossing','galleries','registry')
    panels = ('Source screenshot','Detected backdrop','Finder before','Finder after','Original detail',
              'Before detail','After detail','Removed cloud','Removed blue sky','Removed building')
    column, left, top, step = 300, 208, 138, 264
    page = Image.new('RGB',(left+len(layouts)*column+12, top+step*len(panels)), '#f8f7f1')
    draw = ImageDraw.Draw(page)
    label(draw,(16,12),'Does the Surface Finder retain clouds?',31,bold=True)
    label(draw,(16,60),'Cyan shows blue sky. Pink shows cloud candidates. Green shows predicted surfaces. Red shows removed pixels on recorded building faces.',20)
    for i,name in enumerate(panels):
        label(draw,(15,top+i*step+15),name,20,bold=True)
    views = []
    for j,layout in enumerate(layouts):
        def rank(r):
            n = r['native_pixels']
            return n['removed_cloud'], n['removed_blue'], n['flagged_cloud']
        row = max((r for r in rows if r['layout']==layout),key=rank)
        case = cases[row['key']]
        with Image.open(case.path('image')) as src:
            photo = src.convert('RGB')
        parts = read_gzip(out/row['predictions'])
        blue,cloud = sky_filter.sky_masks(photo)
        before,after = [occupied(parts[m], photo.size) for m in NAMES]
        removed = before & ~after
        building,_,_ = reference_masks(case)
        # Center on the densest changed neighborhood so a few remote pixels cannot shrink the detail.
        focus = removed & cloud
        if not focus.any():
            focus = removed if removed.any() else before
        y,x = np.nonzero(focus)
        bw,bh=240,180
        if len(x):
            cell_width=(photo.width+63)//64
            cell=np.bincount((y//64)*cell_width+x//64).argmax()
            cx,cy=(cell%cell_width)*64+32,(cell//cell_width)*64+32
        else:
            cx,cy=photo.width//2,photo.height//2
        x0,y0=int(np.clip(cx-bw//2,0,photo.width-bw)),int(np.clip(cy-bh//2,0,photo.height-bh))
        box=(x0,y0,x0+bw,y0+bh)
        locator=photo.copy()
        ImageDraw.Draw(locator).rectangle(box,outline='white',width=3)
        old, new = [tint(photo,m,'#3bffc2') for m in (before,after)]
        cells=[locator,tint(tint(photo,blue,'#22dbff'),cloud,'#ff5dba'),old,new,photo.crop(box),
               old.crop(box),new.crop(box),tint(photo,removed&cloud,'#ff5dba').crop(box),
               tint(photo,removed&blue,'#22dbff').crop(box),tint(photo,removed&building,'red').crop(box)]
        x0=left+column*j
        label(draw,(x0,97),layout.title(),22,bold=True)
        n=row['native_pixels']
        for i,img in enumerate(cells):
            y0=top+step*i
            _picture(page,img,x0,y0,column-8,205)
            if i==0:
                label(draw,(x0,y0+210),case.id.split(layout+'-')[-1],16)
                label(draw,(x0,y0+231),case.floor,16)
            if i>=7:
                key=('removed_cloud','removed_blue','removed_recorded_building')[i-7]
                label(draw,(x0,y0+212),f'{n[key]} pixels in full frame',16)
            if i in (2,3):
                m=('current','sky')[i-2]
                s=row['scores'][m]
                label(draw,(x0,y0+212),f"{s['matched']} matched / {s['eligible']} scored",16)
        views.append(dict(layout=layout,frame=row['key'],sha256=row['sha256'],detail_xyxy=box))
    page.save(out/'comparison.jpg',quality=93,subsampling=0,optimize=True)
    save(out/'sheet-views.json',views)


@image_only()
def run(out, workers):
    source=read(BASE/'manifest.json')
    rows=read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    checked(ARCHIVE,source['reference_sha256'])
    cohorts,suite,fixtures=groups()
    cases={f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    roster={r['key']:r['sha256'] for r in rows}
    if (len(rows)!=56 or len(set(roster.values()))!=52 or cases.keys()!=roster.keys() or
            {f['key']:f['sha256'] for f in source['frames']}!=roster or
            source['fixtures']!=fixtures or source['protocol']!=PROTOCOL):
        raise ValueError('Frozen data and scoring protocol differ')
    out.mkdir(parents=True,exist_ok=False)
    code=revision()
    jobs=[(cases[r['key']],r,source['artifacts'][r['predictions']],out) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results=list(pool.map(evaluate,jobs))
    scores={}
    for group in ('all',*COHORTS,*sorted({r['layout'] for r in rows})):
        subset=[r for r in results if group in ('all',r['layout'],r['cohort'])]
        metrics=summary(subset,NAMES)
        metrics['native_pixels']=dict(sum((Counter(r['native_pixels']) for r in subset),Counter()))
        # Counter addition drops zero-valued keys. Retain zeros for explicit reporting.
        for k in results[0]['native_pixels']:
            metrics['native_pixels'].setdefault(k,0)
        scores[group]=metrics
    save(out/'cases.json',results)
    save(out/'scores.json',scores)
    (out/'pictures.md').write_text(picture_table(results,NAMES,NAMES),encoding='utf8')
    render(out,results,cases)
    save(out/'manifest.json',dict(run_id=out.name,code=code,version=suite['version'],frames=source['frames'],
        fixtures=fixtures,reference_sha256=source['reference_sha256'],protocol=PROTOCOL,
        splits=source['splits'],capture=source['capture'],baseline='plane-recovery-crop-splits',methods=NAMES,
        source_run=BASE.relative_to(ROOT).as_posix(),source_manifest_sha256=digest(BASE/'manifest.json'),
        filter_settings=sky_filter.SETTINGS, source_model_timing=source['source_model_timing'],new_inference_s=0,
        selection='Frozen detector applied to both development and previously reviewed regression layouts.',
        timing='Cached gate seconds include sky detection and mask subtraction. They exclude scoring and I/O.',
        native_truth='Recorded building faces at native pixel centers. Upward background uses the recorded camera only for scoring. Blue vs cloud uses detector labels, not cloud ground truth.',
        reproduce='python -m benchmark.sky_round --out runs/sky-replay',artifacts={
            p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(scores['all']['native_pixels'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'sky-{uuid4().hex[:12]}')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    run(args.out.resolve(),args.workers)
