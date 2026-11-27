"""Refresh the weakest-structure viewer from the verified placement exports."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from benchmark import facade_inset_round as suite, roof_span_worst_sheet as sheet
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.edge_vote_outline_eval import recount
from benchmark.face_corner_overlay import corner_pair, edge_overlay
from benchmark.shared_face_error_budget import current_pack, current_stages, measure
from benchmark.shared_face_viewer import write_viewer
from benchmark.strip_visibility_audit import block_footprints
from benchmark.surface_pipeline_sheet import PAPER
from benchmark.surface_sheet import text
from benchmark.texture_camera_eval import checked, read
from benchmark.thin_face_placement_run import ref_inputs
from engine import atlas_count, eave_pinholes, jagged_texture_join, smooth_atlas_outline, smooth_extent_placement
from engine.surface_perimeter import with_outline
from engine.outline_issues import outline_issues
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs

ROOT = suite.ROOT/'benchmark/reviews'
PAGE = ROOT/'shared-face-round-01'
RELEASE = ROOT/'pavement-local-01'
ASSETS = RELEASE/'viewer-overlays'
REVISION = '3b442a3'
SMOOTH_LAYOUTS = {'Shopfront','Registry','Portico','Courtyard'}


@lru_cache(maxsize=1)
def calibrated_atlas():
    return atlas_count.load_atlas(suite.ATLAS,'26.3')


def pre_background_stages(job, expected):
    image, parts, counts, camera = current_stages(job,expected)
    parts,_ = eave_pinholes.refine(image,parts)
    if not any(expected[k] for k in ('returns','texture','outlines')):
        parts,joins,_ = jagged_texture_join.refine(image,parts,'26.3',camera=camera)
        if joins:
            measured = eave_pinholes.measurement_parts([with_outline(p) for p in parts])
            atlas = calibrated_atlas() if 'atlas_phase' in counts else None
            counts = recount(image,measured,atlas)
    parts,_ = smooth_atlas_outline.refine(image,parts,'26.3',atlas_loader=calibrated_atlas)
    return image,parts,counts,camera


# Frozen input to the smooth-wall and counted-corner benchmarks.
latest_stages = pre_background_stages


def structure_stages(job, expected, *, details=False):
    from engine import exterior_foreground, horizon_continuity
    image, original, counts, camera = pre_background_stages(job, expected)
    before_counts = counts
    parts, exterior = exterior_foreground.refine(image, original, camera)
    parts, horizon = horizon_continuity.refine(image, parts)
    excluded = None
    if exterior or horizon:
        parts = [with_outline(p) if 'outline' not in p else p for p in parts]
        measured = eave_pinholes.measurement_parts(smooth_atlas_outline.measurement_parts(parts))
        counts = recount(image, measured, calibrated_atlas() if 'atlas_phase' in counts else None)
        if details:
            def union(patches):
                mask = np.zeros(image.size[::-1], bool)
                for patch in patches:
                    mask |= native_mask(patch, image.size)
                return mask
            excluded = union(original) & ~union(parts)
    stages = image, parts, counts, camera
    return (*stages, excluded, original, before_counts) if details else stages


def read_releases(directories):
    history = defaultdict(list)
    payloads = {}
    for directory in directories:
        release = read(directory/'summary.json')
        with ZipFile(checked(directory/'default-exports.zip',release['archive_sha256'])) as archive:
            for row in release['changed']:
                key = row['id']
                if history[key]:
                    assert history[key][-1]['pack_sha256']==row['previous_pack_sha256']
                history[key].append(row)
                payloads[key] = archive.read(key+'/reconstruction.zip')
                assert sha256(payloads[key]).hexdigest()==row['pack_sha256']
    return release,history,payloads


def regrade(task):
    job,expected,old,updates,payload = task
    args = (job,expected,old['layout'],old['shape_sha256'])
    if not updates:
        return measure(args,stages_fn=structure_stages)
    new=updates[-1]
    assert (old['rgb_sha256'],old['pack_sha256'])==(new['source_hash'],updates[0]['previous_pack_sha256'])
    shift=new['result'].get('translation',old['translation'])
    pack=sorted(decoded_pack(BytesIO(payload))),new['pack_sha256'],shift,new['result']['strict']
    result=measure(args,pack=pack,stages_fn=structure_stages)
    assert all(result['required'][k]==new['result']['facade'][k] for k in ('correct','missed','extras'))
    return result


def atlas_grid_phases(image, parts, counts, atlas):
    """Measure optional atlas offsets using the same source masks and counter as an ordinary build."""
    bare = {r['id']: r for r in counts['surfaces']}
    measured = eave_pinholes.measurement_parts(smooth_atlas_outline.measurement_parts(parts))
    candidates = [p for p in measured if None in bare[p['id']]['phase']
                  and None not in bare[p['id']]['counts'] and bare[p['id']].get('quad')]
    if not candidates:
        return {}
    phases = {}
    for row in recount(image, candidates, atlas)['surfaces']:
        old = bare[row['id']]
        if old['phase'] == row['phase']:
            continue
        # Both modes must describe the same measured chart. Preserve each public
        # dimension (including unconfirmed inset hosts) and every measured offset.
        assert old['quad'] == row['quad'] and old['counts'] == row['counts'], row['id']
        assert all(a is None or a == b for a, b in zip(old['phase'], row['phase'])), row['id']
        phases[row['id']] = row['phase']
    return phases


def f1(row):
    return 2*row['correct'] / (2*row['correct']+row['missed']+row['extras'])


def gap_contribution(rows, pool):
    errors = sum(r['required']['missed']+r['required']['extras'] for r in rows)
    total_errors = pool['missed']+pool['extras']
    return dict(views=len(rows), errors=errors, points=100*errors/(2*pool['correct']+total_errors),
                share=100*errors/total_errors)


def totals(group):
    return {k:sum(r['required'][k] for r in group) for k in ('correct','missed','extras')}


def target_progress(group):
    # Rearrange 2*(C+m)/(2*C+M+E+m-e) >= .95 for corrections m and e.
    g = totals(group)
    return 105*g['missed']+95*g['extras']


def themes(group):
    def aggregate(key, required=True):
        result = Counter()
        for row in group:
            result.update(row[key].get('required',{}) if required else row[key])
        return result
    projection = aggregate('missing_projection')
    uncovered = aggregate('uncovered_stage')
    return dict(uncovered=projection['uncovered'], position=projection['same_axis'],
                direction=projection['different_axis'], ties=projection['coverage_tie']+projection['axis_tie'],
                noMask=uncovered['no_mask'], maskAxis=uncovered['mask_without_correct_axis'],
                count=uncovered['correct_axis_no_dimensions'], origin=uncovered['dimensions_no_origin'],
                background=aggregate('extra_projection',False)['mostly_on_background'])


def path_to_target(groups, pool, *, stop_at_target=True):
    remaining = pool.copy()
    result = []
    for group in groups:
        g = totals(group)
        remaining['correct'] += g['missed']
        remaining['missed'] -= g['missed']
        remaining['extras'] -= g['extras']
        # Lead with the view contributing the most to reaching 95%.
        views = sorted(group,key=lambda r:(-target_progress([r]),r['id']))
        result.append(dict(name=group[0]['layout'], ids=[r['id'] for r in views],
                           missed=g['missed'],extras=g['extras'],score=100*f1(remaining),themes=themes(group)))
        if stop_at_target and result[-1]['score']>=95:
            break
    return result


def required_overlay(image, labels, ids, missing, metric):
    owners, _, _, bits, square, _ = block_footprints(labels,ids)
    classes = np.zeros(len(bits), np.uint8)
    for block in labels['blocks']:
        if tuple(block['xyz_world']) in missing:
            i = block['id']
            classes[i] = 2 if bits[i] in (1,2) and square[i]<4 else 1
    assert (classes==1).sum() == metric['missed']
    pixels = classes[owners[ids]]
    rgb = np.asarray(image).copy()
    for kind, color in [(1,(33,148,255)),(2,(255,188,40))]:
        active = pixels==kind
        rgb[active] = .25*rgb[active]+.75*np.asarray(color)
    return Image.fromarray(rgb)


def main():
    release,changed,payloads = read_releases([RELEASE,ROOT/'smooth-local-placement-01',ROOT/'smooth-joint-placement-01',
                                             ROOT/'window-wall-01',ROOT/'counted-ground-corner-01',ROOT/'workshop-foreground-01'])
    fixed = read(PAGE/'error-budget.json')['frames']
    jobs = {j[1]['id']:j for j in suite.jobs(suite.framed.EVALUATION)}
    assert len(fixed)==len(jobs)==104 and {r['id'] for r in fixed}==set(jobs)
    inputs = {r['id']:r for r in read(ROOT/'facade-inset-round-01/combined.json')['rows']}
    corners = {r['id']:r for r in read(ROOT/'counted-ground-corner-01/cases.json')}
    by_shape = defaultdict(list)
    with ProcessPoolExecutor(4) as pool:
        tasks=[(jobs[r['id']],inputs[r['id']],r,changed.get(r['id']),payloads.get(r['id'])) for r in fixed]
        for r,observed in zip(fixed,pool.map(regrade,tasks),strict=True):
            r.update(observed)
            by_shape[r['shape_sha256']].append(r)
    for metric, name in [('strict','strict'),('required','facade')]:
        for k in ('correct','missed','extras'):
            assert sum(r[metric][k] for r in fixed) == release['cohorts']['fixed'][name][k]
    def order(row):
        return f1(row['strict']), row['id']
    selected = sorted((min(group,key=order) for group in by_shape.values()),key=order)[:10]
    facade_pool = release['cohorts']['fixed']['facade']
    layout_views = [v for r in selected for v in by_shape[r['shape_sha256']]]
    gap = dict(total=100*(1-f1(facade_pool)), selected=gap_contribution(selected,facade_pool),
               layouts=gap_contribution(layout_views,facade_pool))
    paths = dict(layouts=path_to_target(sorted(by_shape.values(),key=target_progress,reverse=True),facade_pool),
                 images=path_to_target(sorted(([r] for r in fixed),key=target_progress,reverse=True),facade_pool),
                 displayed=path_to_target([by_shape[r['shape_sha256']] for r in selected],facade_pool,stop_at_target=False))
    extras = {key for path in paths.values() for row in path for key in row['ids']}
    # Keep the investigated layouts selectable even after a repair moves them out of the lowest ten.
    review_layouts = SMOOTH_LAYOUTS | {'Workshop', 'Galleries'}
    extras.update(r['id'] for r in fixed if r['layout'] in review_layouts)
    ids = {r['id'] for r in selected}
    display = selected + sorted((r for r in fixed if r['id'] in extras-ids),key=lambda r:(r['layout'],r['id']))
    gap.update(paths=paths, baseline=100*f1(facade_pool), themes=themes(fixed),
               displayedCeiling=path_to_target([selected],facade_pool)[0]['score'])
    atlas = atlas_count.load_atlas(suite.ATLAS,'26.3')
    rows = []
    ASSETS.mkdir(exist_ok=True)
    left, width, top, height = 201, 415, 112, 451
    board = Image.new('RGB',(left+width*len(selected),top+10*height),PAPER)
    pen = ImageDraw.Draw(board)
    text(pen,14,9,'Ten lowest-scoring structures by their weakest fixed view',29,True)
    text(pen,15,50,'Default '+REVISION+' · Strict relative placement F1 · 104 whole-structure source images',19,False)
    names = ['Source image','Surface Finder','Surface crop','Straightened crop','Block Counter','Grid on source image',
             'Material sample','Projected build','Visible reference','Placement errors']
    for i,name in enumerate(names):
        text(pen,10,top+i*height+15,name,18,True)
    for index,r in enumerate(display):
        key, strict, facade = r['id'], r['strict'], r['required']
        job = jobs[key]
        _, frame, _, control = job
        assert frame['files']['rgb.png']==r['rgb_sha256']
        if key in changed:
            payload = payloads[key]
            assert sha256(payload).hexdigest()==r['pack_sha256']
            cells = decoded_pack(BytesIO(payload))
        else:
            cells, pack_hash, _, _ = current_pack(job)
            assert pack_hash==r['pack_sha256']
        image, parts, counts, camera, excluded, original, before_counts = structure_stages(job,inputs[key],details=True)
        _,_,_,_,pose,labels,ids,truth,_,_,states = sheet.stages.read_view(suite.CONTROL,*job)
        _,_,transform,_ = ref_inputs(job)
        raw = registered(cells,transform)
        scored = suite.framed.spacing.shift(raw,r['translation'])
        assert (len(scored&truth),len(truth-scored),len(scored-truth)) == tuple(strict[k] for k in ('correct','missed','extras'))
        group = by_shape[r['shape_sha256']]
        info = dict(name=r['layout'],rank=index+1,shape_sha256=r['shape_sha256'],frames=len(group),
                    f1=f1({k:sum(a['strict'][k] for a in group) for k in ('correct','missed','extras')}),
                    representative=dict(id=key,rgb_sha256=r['rgb_sha256'],pack_sha256=r['pack_sha256'],f1=f1(strict)))
        geom = dict(strict, placed=len(scored),target=len(truth))
        current = dict(control,relative_geometry=geom,translation=r['translation'],
                       span=dict(control['span'],after=(np.ptp(list(raw),axis=0)+1).tolist() if raw else [0]*3))
        if key in suite.framed.roof_specs():
            current['roof'] = suite.framed.roof(frame,raw,truth)
        view = (image,parts,counts['surfaces'],camera,pose,labels,ids,truth,raw,scored,states)
        atlas_phases = atlas_grid_phases(image, parts, counts, atlas)
        panels,_ = sheet.prepare_view(info,current,atlas,view,grid_phases=atlas_phases)
        prefix = f'{index+1:02}'
        if atlas_phases:
            panels[-1][0].save(ASSETS/f'{prefix}-grid-atlas.jpg',quality=86,optimize=True)
        surface, wire, visible_faces = edge_overlay(image.size, parts, camera, cells)
        for overlay, suffix in ((surface, 'mask-edges'), (wire, 'face-corners')):
            overlay.save(ASSETS/f'{prefix}-{suffix}.png', transparency=0, optimize=True)
        excluded_pixels = int(excluded.sum()) if excluded is not None else 0
        if excluded_pixels:
            mask = Image.fromarray(excluded.astype(np.uint8))
            mask.putpalette([0, 0, 0, 184, 119, 255])
            mask.save(ASSETS/f'{prefix}-excluded.png', transparency=0, optimize=True)
        for stage,i in [('source',0),('outlines',1),('grid',5),('build',7),('reference',8),('errors',9)]:
            panels[i][0].save(ASSETS/f'{prefix}-{stage}.jpg',quality=86,optimize=True)
        required_overlay(image,labels,ids,truth-scored,facade).save(ASSETS/f'{prefix}-required.jpg',quality=89,optimize=True)
        if index<len(selected):
            col = left+index*width
            text(pen,col,78,f"{index+1}. {r['layout']}   {f1(strict):.2%}",20,True)
            for i,(pic,crop,caption) in enumerate(panels[:10]):
                sheet.panel(board,(col,top+i*height,width-12,height-14),(pic,crop,caption.replace('Photo / atlas','Source / atlas')))
        smooth=None
        if r['layout'] in SMOOTH_LAYOUTS:
            smooth_parts=smooth_atlas_outline.measurement_parts(parts)
            proposed=np.zeros((image.height,image.width),bool)
            for shared in (False,True):
                for region in smooth_extent_placement.proposals(image,smooth_parts,common_wall=shared):
                    for y,a,b in region['runs']:
                        proposed[y,a:b]=True
            spans=row_runs(proposed)
            accepted=np.zeros((image.height,image.width),bool)
            for p in parts:
                if 'measured_smooth_support' in p:
                    old_mask=native_mask(dict(support=p['measured_smooth_support']),image.size)
                    accepted |= native_mask(p,image.size) & ~old_mask
            approved=row_runs(accepted)
            assert not (accepted & ~proposed).any(),key
            smooth=dict(pixels=sum(b-a for y,a,b in spans),
                        path=''.join(f'M{a} {y+.5}h{b-a}' for y,a,b in spans),
                        accepted=int(accepted.sum()),acceptedPath=''.join(f'M{a} {y+.5}h{b-a}' for y,a,b in approved))
        rows.append(dict(id=key,name=r['layout'],prefix=prefix,issues=outline_issues(parts,image.size),smooth=smooth,
                         visibleFaces=visible_faces,cornerPair=corner_pair(image,original,before_counts,camera,corners[key]),
                         excludedPixels=excluded_pixels,atlasGridSurfaces=len(atlas_phases),
                         f1=f'{f1(strict):.2%}',facade=f'{f1(facade):.2%}',
                         correct=strict['correct'],required=facade['missed'],optional=strict['missed']-facade['missed'],
                         extras=strict['extras'],gap=gap_contribution([r],facade_pool),
                         layoutGap=gap_contribution(group,facade_pool),themes=themes([r])))
    board.save(ASSETS/'stage-map.jpg',quality=91,subsampling=0,optimize=True)
    write_viewer(PAGE,rows,'../pavement-local-01/viewer-overlays',REVISION,gap)


if __name__=='__main__':
    main()
