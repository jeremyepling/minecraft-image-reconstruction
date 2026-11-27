"""Grade the frozen visible-surface masks before and after backdrop cleanup."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np

from benchmark import current_placement_viewer as viewer
from benchmark.mask_blocks import decoded_pack
from benchmark.wall_texture_evaluate import metrics
from benchmark.texture_camera_eval import digest, read, save
from engine.pillar_seams import native_mask
from engine.horizon_continuity import refine

OUT = viewer.ROOT/'workshop-horizon-01'


@lru_cache(maxsize=1)
def inputs():
    return {row['id']:row for row in read(viewer.ROOT/'facade-inset-round-01/combined.json')['rows']}


@lru_cache(maxsize=1)
def foreground_module(source):
    if source is None:
        from engine import exterior_foreground
        return exterior_foreground
    spec = importlib.util.spec_from_file_location('exterior_foreground', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grade(task):
    job, combined, source = task
    frame = job[1]
    baseline = getattr(viewer, 'pre_background_stages', viewer.latest_stages)
    image, parts, _, camera = baseline(job, inputs()[frame['id']])
    first, exterior = (foreground_module(source).refine(image,parts,camera) if combined else (parts, []))
    filtered, horizon = refine(image, first)
    touched = {event['id'] for event in exterior + horizon}
    row = dict(id=frame['id'], split=job[0], source_hash=frame['files']['rgb.png'], events=[],
               exterior=exterior, horizon=horizon)
    if not touched:
        return row
    truth = viewer.suite.visible_map(*job[:2])
    before = {p['id']:p for p in parts}
    after = {p['id']:p for p in filtered}
    all_removed = np.zeros(image.size[::-1], bool)
    for key in sorted(touched):
        mask = native_mask(before[key], image.size)
        if key in after:
            mask &= ~native_mask(after[key], image.size)
        all_removed |= mask
        foreground = int(np.count_nonzero(truth[mask]))
        row['events'].append(dict(id=key, foreground=foreground, background=int(mask.sum())-foreground,
                                  wholly_removed=key not in after))
    row['unique_removed'] = int(all_removed.sum())
    a = metrics(parts, truth, image.size)
    b = metrics(filtered, truth, image.size)
    row.update(matches_gained=sorted(set(b['matched'])-set(a['matched'])),
               matches_lost=sorted(set(a['matched'])-set(b['matched'])),
               change_paired_pixels=b['paired_pixels']-a['paired_pixels'],
               change_unmatched=b['unmatched_predictions']-a['unmatched_predictions'])
    return row


def summarize(rows):
    def metrics(group):
        events = [e for r in group for e in r['events']]
        return dict(frames=len(group), changed=sum(bool(r['events']) for r in group),
                    masks=len(events), wholly_removed=sum(e['wholly_removed'] for e in events),
                    foreground=sum(e['foreground'] for e in events), background=sum(e['background'] for e in events),
                    unique_removed=sum(r.get('unique_removed',0) for r in group),
                    gained=sum(len(r.get('matches_gained',[])) for r in group),
                    lost=sum(len(r.get('matches_lost',[])) for r in group),
                    paired_pixel_delta=sum(r.get('change_paired_pixels',0) for r in group),
                    unmatched_delta=sum(r.get('change_unmatched',0) for r in group))
    return {name:metrics([r for r in rows if selected(r)]) for name,selected in [
        ('fixed',lambda r:r['split'] in viewer.suite.framed.EVALUATION),
        ('other',lambda r:r['split'] not in viewer.suite.framed.EVALUATION),
        ('workshop',lambda r:'workshop' in r['id']),('galleries',lambda r:'galleries' in r['id']),
        ('shopfront',lambda r:'shopfront' in r['id'])]}


def verify_ordinary(folder, jobs):
    manifest = read(folder/'build-manifest.json')
    assert manifest['through'] == 'placement'
    job = next(j for j in jobs if j[1]['files']['rgb.png'] == manifest['image_sha256'])
    key = job[1]['id']
    old_stages = getattr(viewer, 'pre_background_stages', viewer.latest_stages)
    image, original, previous_counts, _ = old_stages(job, inputs()[key])
    expected, events = refine(image, original)
    assert events == manifest['surface_finder']['boundary_settings']['horizon_continuity']['events']
    visible = {part['id']:part for part in read(folder/'surfaces.json')['planes']}
    for part in expected:
        record = visible[part['id']]
        mask = native_mask(part, image.size)
        assert np.array_equal(native_mask(record, image.size),mask)
        assert (folder/record['source_crop']['path']).is_file()
        assert (folder/record['context_crop']['path']).is_file()
        y, x = np.nonzero(mask)
        assert record['source_crop']['xyxy'] == [x.min(),y.min(),x.max()+1,y.max()+1]
    reported = read(folder/'counts.json')
    assert set(visible) == {p['id'] for p in expected} == {r['id'] for r in reported['surfaces']}
    previous = {r['id']:r for r in previous_counts['surfaces']}
    for r in reported['surfaces']:
        assert r['counts'] == previous[r['id']]['counts'] and r['dimensions'] == previous[r['id']]['dimensions']
        if 'sample' in r:
            assert (folder/r['sample']).is_file()
    assert reported.get('block_groups') == previous_counts.get('block_groups')
    _, _, archives = viewer.read_releases([viewer.RELEASE,viewer.ROOT/'smooth-local-placement-01',
        viewer.ROOT/'smooth-joint-placement-01',viewer.ROOT/'window-wall-01',viewer.ROOT/'counted-ground-corner-01'])
    old = (decoded_pack(BytesIO(archives[key])) if key in archives
           else dict.fromkeys(viewer.current_pack(job)[0], 'minecraft:smooth_stone'))
    assert decoded_pack(folder/'reconstruction.zip') == old
    save(OUT/'ordinary.json',dict(id=key,source_hash=manifest['image_sha256'],events=events,
        visible_surfaces=len(visible),all_source_crops_verified=True,all_reported_dimensions_verified=True,
        all_material_groups_verified=True,verified_blocks=len(old),pack_sha256=digest(folder/'reconstruction.zip')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combined', action='store_true')
    parser.add_argument('--foreground-source', type=Path, help='Read-only module in a separate experiment checkout')
    parser.add_argument('--ordinary', type=Path, help='Verify a horizon-only public placement build')
    args = parser.parse_args()
    jobs = list(viewer.suite.jobs([*viewer.suite.framed.SIZES,'shelter']))
    if args.ordinary:
        verify_ordinary(args.ordinary, jobs)
        return
    with ProcessPoolExecutor(5) as pool:
        rows = list(pool.map(grade, ((job,args.combined,args.foreground_source) for job in jobs)))
    assert len(rows)==204
    prefix = 'combined-' if args.combined else ''
    OUT.mkdir(exist_ok=True,parents=True)
    cases = OUT/(prefix+'cases.json')
    save(cases, rows)
    summary = summarize(rows)
    save(OUT/(prefix+'summary.json'),dict(baseline='53252df',cases_sha256=digest(cases),cohorts=summary))
    print(summary, flush=True)


if __name__=='__main__':
    main()
