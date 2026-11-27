"""Verify the 204 paired exports and the independent two-shot RGB CLI comparison."""
import argparse
from collections import Counter
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from benchmark import framed_depth_eval as f
from benchmark.guarded_wall_outlines_trial import _measure
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import STABLE, checked, digest, read, save
from benchmark.wall_texture_evaluate import metrics
from engine.surface_perimeter import _native_alpha


def facade(before, after, primary):
    from engine.atlas_count import load_atlas
    from engine.surface_model import _count_regions
    from engine.texture_guided_count import METHOD
    key = 'heldout-facade-oblique-right'
    frame = next(row for row in f.cases(['native']) if row[1]['id']==key)
    truth_row = next(row for row in read(primary/'cases.json') if row['id']==key)
    image, _, _, _, _, labels, ids, *_ = read_view(primary, *frame, truth_row)
    faces = labels['faces']
    lut = np.zeros(len(faces)+1, int)
    for face in faces:
        lut[face['id']] = face['surface_id']
    truth = lut[ids]
    manifests = [read(d/'build-manifest.json') for d in (before,after)]
    assert manifests[0]['image_sha256'] == manifests[1]['image_sha256'] == frame[1]['files']['rgb.png']
    assert manifests[0]['model_sha256'] == manifests[1]['model_sha256']
    event, = manifests[1]['surface_finder']['boundary_settings']['diagonal_wall_join']['events']
    atlas = load_atlas(STABLE/'runs/26.3/atlas','26.3')
    views = []
    controls = []
    for path, changed in [(before,event['inputs']), (after,[event['output']])]:
        parts = read(path/'surfaces.json')['planes']
        counts = read(path/'counts.json')
        with TemporaryDirectory() as tmp:
            with_atlas = _count_regions(image, parts, '26.3', METHOD, atlas, Path(tmp))
        positions = read(path/'placements.json')['blocks']
        pack = decoded_pack(path/'reconstruction.zip')
        assert {tuple(b[k] for k in 'xyz'):'minecraft:'+b['block_id'] for b in positions} == pack
        controls.append(pack)
        ids_out = {p['id'] for p in parts}
        for item in parts:
            assert all(j['neighbor'] in ids_out and j['neighbor']!=item['id'] for j in item.get('corner_junctions',[]))
        m = metrics(parts,truth,image.size)
        best = max((r for r in m['parts'] if r['id'] in changed),key=lambda r:r['correct_if_dominant'])
        sid = best['dominant']
        target = int(np.count_nonzero(truth==sid))
        crop_stats = []
        for item in parts:
            if item['id'] not in changed:
                continue
            crop = item['source_crop']
            alpha, box = _native_alpha(item['support'], image.size)
            assert list(box)==crop['xyxy']
            with Image.open(path/crop['path']) as png:
                assert np.array_equal(np.asarray(png.getchannel('A')), np.asarray(alpha))
                visible = np.asarray(alpha,bool)
                assert np.array_equal(np.asarray(png.convert('RGB'))[visible],np.asarray(image.crop(box))[visible])
            assert item['sampling_chart']==item.get('quad')
            crop_stats.append(dict(id=item['id'], xyxy=list(box), observed_pixels=int(visible.sum())))
        hits = best['correct_if_dominant']
        views.append(dict(proposals=len(parts), matched=len(m['matched']), target=m['target'],
                          unmatched=m['unmatched_predictions'], paired_pixels=m['paired_pixels'],
                          background_pixels=m['background_pixels'], surface=sid, source_pixels=target,
                          recall=hits/target,iou=hits/(target+best['pixels']-hits), parts=crop_stats,
                          count_rows=[_measure(c) for c in counts['surfaces'] if c['id'] in changed],
                          material_groups=len(counts['block_groups']), placed_materials=dict(Counter(pack.values())),
                          explicit_atlas_groups=with_atlas['material_block_counts'], explicit_atlas_phase=with_atlas['atlas_phase'],
                          surfaces_sha256=digest(path/'surfaces.json'), counts_sha256=digest(path/'counts.json'),
                          placements_sha256=digest(path/'placements.json'), export_sha256=digest(path/'reconstruction.zip')))
    assert controls[0]==controls[1]
    assert views[0]['surface']==views[1]['surface']
    recorded = [a for a in faces if a['surface_id']==views[0]['surface']]
    xyz = np.array([a['xyz_world'] for a in recorded])
    assert len(set(a['direction'] for a in recorded))==1
    return dict(frame=key, rgb_sha256=frame[1]['files']['rgb.png'],models=manifests[1]['model_sha256'],
                event=event,before=views[0],after=views[1],
                visible_surface=dict(direction=recorded[0]['direction'],minimum=xyz.min(0).tolist(),
                                    maximum=xyz.max(0).tolist(),visible_unit_faces=len(recorded)))


def main(root, control, out):
    out.mkdir(parents=True, exist_ok=True)
    evidence = []
    for cohort, source, comparison in [('primary','primary-verified','primary-production'),
                                        ('shelter','shelter-verified','transfer-production')]:
        path = root/source
        rows = read(checked(path/'cases.json',read(path/'summary.json')['cases_sha256']))
        for row in rows:
            split, key = row.get('split','shelter'), row['id']
            rel = (Path(split)/key if cohort=='primary' else Path(key)) / 'reconstruction.zip'
            old = decoded_pack(checked(control/comparison/rel,row['control_pack_sha256']))
            new = decoded_pack(checked(path/rel, row['pack_sha256']))
            events = row['status']['events']
            evidence.append(dict(split=split,frame=key,rgb_sha256=row['rgb_sha256'],
                                 control_pack_sha256=row['control_pack_sha256'],candidate_pack_sha256=row['pack_sha256'],
                                 joined='|'.join(','.join(e['inputs']) for e in events),
                                 joined_native_pixels=sum(e['output_area_px'] for e in events),
                                 same_decoded_blocks_and_materials=old==new))
    assert len(evidence)==204 and len({(r['split'],r['frame']) for r in evidence})==204
    with (out/'frames.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=evidence[0])
        writer.writeheader()
        writer.writerows(evidence)
    summary = read(root/'primary-verified/summary.json')
    save(out/'result.json',dict(control=summary['control'],candidate=summary['revision'],views=len(evidence),
        affected=sum(bool(row['joined']) for row in evidence),
        identical_exports=sum(row['same_decoded_blocks_and_materials'] for row in evidence),
        frames_sha256=digest(out/'frames.csv'),
        facade_priority=summary['facade_priority'],
        cli=facade(root/'control-cli-facade',root/'verified-cli-facade',control/'primary-production')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--control-runs',type=Path,default=Path('../visible-error-budget/runs'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/guarded-surface-outlines-01'))
    args=parser.parse_args()
    main(args.runs,args.control_runs,args.out)
