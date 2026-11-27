"""Check opening filters on recorded platform tops and same-pavement counterfactuals."""
import argparse
from pathlib import Path
import importlib.util

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark import facade_inset_round as suite
from benchmark.texture_camera_eval import digest, save
from benchmark.shared_face_error_budget import current_pack
from benchmark.thin_face_placement_run import ref_inputs
from benchmark.mask_blocks import registered
from benchmark.texture_camera_sheet import font
from engine.pillar_seams import native_mask, orientation
from engine.plane_consensus import row_runs


REVIEW = Path(__file__).resolve().parent / 'reviews/pavement-platform-controls-01'
COLONNADE = 'inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right'
APERTURE = 'planarity-new-14-0'
SLIT = 'planarity-new-18-1'


def authored_platform(frame, labels):
    """Named authored fixture parts, identified by their height and material."""
    result = []
    blocks = {b['id']: b['block_id'].split(':')[-1] for b in labels['blocks']}
    for face in labels['faces']:
        if face['direction'] != 'up':
            continue
        height = face['xyz_world'][1] - frame['ground_y']
        material = blocks[face['block']]
        family = frame['family']
        if (family == 'loadingdock' and (height, material) == (2, 'stone_bricks') or
                family == 'platformshelter' and (height, material) == (2, 'cut_copper') or
                family == 'switchback' and (material in ('bricks', 'polished_andesite') and 1 <= height <= 8
                                            or material == 'birch_planks' and height in (4, 8))):
            result.append(face)
    return result


def measured(detect, job):
    split, frame, _, _ = job
    image, parts, _, cam = suite.ordinary_inputs(*job)
    # The candidate sees only the saved source image, masks, and estimated camera.
    records = detect(image, parts, cam)
    reject = {r['id'] for r in records if r['reject']}
    evaluated = {r['id'] for r in records}
    labels, regions = suite.visible_map(split, frame, metadata=True)
    faces = authored_platform(frame, labels)
    assert faces
    ids = {f['surface_id'] for f in faces}
    # Check every selected region belongs wholly to a named top. This permits
    # using the surface-index map in place of the heavier face-index map.
    assert all(f in faces for f in labels['faces'] if f['surface_id'] in ids)
    target = np.isin(regions, sorted(ids))
    masks = {p['id']: native_mask(p, image.size) for p in parts}
    covered = np.zeros_like(target)
    overlaps = {}
    for name, mask in masks.items():
        covered |= mask
        pixels = int((target & mask).sum())
        if pixels:
            overlaps[name] = pixels
    return dict(frame=frame['id'], source_sha256=frame['files']['rgb.png'],
                block_faces=len(faces), named_surface_ids=sorted(ids), platform_pixels=int(target.sum()),
                platform_pixels_in_input_masks=int((target & covered).sum()), overlaps=overlaps,
                candidate_surfaces_with_platform_pixels=sorted(evaluated & set(overlaps)),
                removed_platform_pixels=int(sum(overlaps.get(k, 0) for k in reject)),
                removed_masks=sorted(reject))


def plane(image, cam):
    p, _ = orientation(cam, image.size)
    inv = np.linalg.inv(p[:, :3])
    eye = -inv @ p[:, 3]
    y, x = np.indices(image.size[::-1])
    rays = np.stack((x+.5, y+.5, np.ones_like(x)), axis=-1) @ inv.T
    return eye, rays


def xyz_on_plane(eye, rays, height):
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (height-eye[1]) / rays[:, :, 1]
    return eye + depth[:, :, None]*rays, depth


def supported(part, name, mask):
    return dict(part, id=name, support=dict(size=list(mask.shape[::-1]), runs=row_runs(mask)),
                support_area=int(mask.sum()))


def candidate_record(detect, image, parts, cam, target):
    record = next((r for r in detect(image, parts, cam) if r['id']==target), None)
    return dict(candidate_reached_phase_check=bool(record), suppressed=bool(record and record['reject']),
                rim=record['rim'] if record else [], folds=record['folds'] if record else [])


def ground_tile(image, xyz, visible):
    """Independent native-pixel template for rendering raised control tiles."""
    floor = xyz[visible][:, [0, 2]]
    col = (np.floor(16*floor).astype(int) % 16)
    uv = 16*col[:, 1]+col[:, 0]
    pixels = np.asarray(image)[visible]
    assert len(np.unique(uv))==256
    return np.array([np.median(pixels[uv==i], axis=0) for i in range(256)], np.uint8)


def counterfactuals(detect, local_veto, job):
    image, parts, _, cam = suite.ordinary_inputs(*job)
    masks = {p['id']: native_mask(p, image.size) for p in parts}
    seed = next(p for p in parts if p['id']==APERTURE)
    opening = masks[APERTURE]
    reference = suite.visible_map(job[0], job[1])
    assert not reference[opening].any()
    eye, rays = plane(image, cam)
    ground, depth = xyz_on_plane(eye, rays, 0)
    y, x = np.indices(opening.shape)
    independent = (x<350) & (depth>15) & (depth<80)
    tile = ground_tile(image, ground, independent)
    records = [dict(control='actual-opening', input_pixels=int(opening.sum()), expected='suppress',
                    **candidate_record(detect, image, parts, cam, APERTURE))]
    panels = [('Actual opening', image, opening)]
    cells, _, shift, _ = current_pack(job)
    assert list(shift)==[0,0,0]
    truth, _, transform, _ = ref_inputs(job)
    baseline = set(cells)
    known_correct = {cell for cell in baseline if registered({cell}, transform) <= truth}

    def check_geometry(record, stage_parts, pad_cells, *, clean=False):
        accepted = [p for p in stage_parts if p['id']==APERTURE] if record['suppressed'] else []
        original = (known_correct if clean else baseline) | pad_cells
        removed = local_veto(image.size, cam, accepted, original)
        # Forced selection is a stress diagnostic, separate from the working rule.
        forced = local_veto(image.size, cam, [p for p in stage_parts if p['id']==APERTURE], original)
        return dict(pad_cells=len(pad_cells), pad_removed=len(removed & pad_cells),
                    old_correct_removed=len(removed & known_correct), other_removed=len(removed-known_correct-pad_cells),
                    pad_removed_if_phase_overridden=len(forced & pad_cells))

    records[0]['local']=check_geometry(records[0],parts,set())
    # Re-render the SAME opening footprint with the top of an elevated platform
    # wearing the same 16x16 source pavement texture. Occluding surfaces and the
    # camera stay fixed, and the material now lies on its occupied y=3 plane.
    raised, _ = xyz_on_plane(eye, rays, 3)
    uv = np.floor(16*raised[opening][:, [0,2]]).astype(int)%16
    raised_rgb = np.array(image)
    raised_rgb[opening] = tile[16*uv[:,1]+uv[:,0]]
    raised_image = Image.fromarray(raised_rgb)
    records.append(dict(control='enclosed-raised-same-pavement', surface_height=3, expected='retain',
                        input_pixels=int(opening.sum()), **candidate_record(detect, raised_image, parts, cam, APERTURE)))
    raised_cells = {(int(np.floor(a)),2,int(np.floor(b))) for a,b in raised[opening][:,[0,2]]}
    records[-1]['local']=check_geometry(records[-1],parts,raised_cells,clean=True)
    panels.append(('Raised top in opening', raised_image, opening))

    # Add a finite, square-grid horizontal floor BEHIND the colonnade. Its
    # dark one-block trim can be seen through the opening and beyond the posts.
    # The interior is the actual pavement material at its true ground phase.
    # The same plane and local camera coordinates define the footprint for all
    # visible source rays, including those outside the roof and posts.
    ground_x, ground_z = ground[:,:,0], ground[:,:,2]
    pad = (depth>0) & (ground_x>=2) & (ground_x<18) & (ground_z>=15) & (ground_z<35)
    inside = (ground_x>=3) & (ground_x<17) & (ground_z>=16) & (ground_z<34)
    border = pad & ~inside & (reference==0)
    genuine = pad & inside & (reference==0)
    pixels = np.array(image)
    pixels[border] = (np.asarray(image)[border] * np.array([.63,.36,.21])).astype(np.uint8)
    framed = Image.fromarray(pixels)
    target = opening & genuine
    # Keep the detector's actual aperture segmentation, clipped to the real
    # pad interior. Supply the remainder of the real pad and trim as separate
    # horizontal masks so all available platform surfaces are represented.
    extra_pad = genuine & ~opening
    trimmed_parts = [supported(seed, APERTURE, target) if p['id']==APERTURE else p for p in parts]
    trimmed_parts += [supported(seed, 'pad-outside-aperture', extra_pad),
                      supported(seed, 'pad-edging', border)]
    records.append(dict(control='enclosed-flush-bordered-pavement', expected='retain', size_blocks=[16,20],
                        surface_height=0, input_pixels=int(target.sum()), border_pixels=int(border.sum()),
                        visible_border_inside_aperture=int((border & opening).sum()),
                        pad_pixels_visible_elsewhere=int(extra_pad.sum()),
                        **candidate_record(detect, framed, trimmed_parts, cam, APERTURE)))
    pad_cells={(x,-1,z) for x in range(2,18) for z in range(15,35)}
    records[-1]['local']=check_geometry(records[-1],trimmed_parts,pad_cells)
    panels.append(('Flush bordered floor', framed, target))

    # Independent 3x3 flush, unobstructed pad with supplied input mask.
    xx, zz = ground_x, ground_z
    top = (depth>0) & (xx>=-1) & (xx<2) & (zz>=-7) & (zz<-4)
    assert top.sum()>1000 and not reference[top].any()
    tag='flush-pad'
    new_parts = parts+[supported(seed,tag,top)]
    records.append(dict(control='unenclosed-flush-3x3', size_blocks=[3,3], surface_height=0,
                        input_pixels=int(top.sum()), expected='retain',
                        **candidate_record(detect, image, new_parts, cam, tag)))
    cells3={(x,-1,z) for x in range(-1,2) for z in range(-7,-4)}
    accepted=[new_parts[-1]] if records[-1]['suppressed'] else []
    removed=local_veto(image.size,cam,accepted,baseline|cells3)
    records[-1]['local']=dict(pad_cells=len(cells3),pad_removed=len(removed&cells3),
                              pad_removed_if_phase_overridden=len(local_veto(image.size,cam,[new_parts[-1]],baseline|cells3)&cells3))
    # Contaminate the opening mask with the nearest two pixels of actual
    # roof/post/sill. The RGB and 3D build remain the captured originals.
    grow = np.asarray(Image.fromarray(opening).convert('L').filter(ImageFilter.MaxFilter(5)),bool)
    contamination = grow & (reference>0)
    mixed=opening|contamination
    mixed_parts=[supported(seed,APERTURE,mixed) if p['id']==APERTURE else
                 supported(p,p['id'],masks[p['id']]&~contamination) for p in parts]
    mixed_record=candidate_record(detect,image,mixed_parts,cam,APERTURE)
    mixed_record.update(control='opening-spilling-onto-real-neighbors', input_pixels=int(mixed.sum()),
                        real_foreground_pixels=int(contamination.sum()),expected='preserve real neighbors')
    mixed_record['local']=check_geometry(mixed_record,mixed_parts,set())
    records.append(mixed_record)
    return records, panels


def write_comparison(panels, out):
    sheet = Image.new('RGB', (600*len(panels), 590), '#f8f5eb')
    draw = ImageDraw.Draw(sheet)
    for ix,(name, image, mask) in enumerate(panels):
        left = ix*600
        draw.text((left+14,10), name, fill='#192d32', font=font(24))
        full=image.resize((565,353), Image.Resampling.LANCZOS)
        sheet.paste(full,(left+14,49))
        # The three crops use precisely the same source coordinates.
        crop=image.crop((740,410,900,505)).resize((311,177), Image.Resampling.NEAREST)
        sheet.paste(crop,(left+14,407))
        crop=Image.fromarray(np.uint8(mask)*255).crop((740,410,900,505)).resize((249,148),Image.Resampling.NEAREST)
        sheet.paste(crop,(left+332,430))
        draw.text((left+332,406),'Tested top mask',font=font(17),fill='#192d32')
    sheet.save(out, quality=91)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--local-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, default=REVIEW)
    args = p.parse_args()
    path = args.candidate_root / 'engine/pavement_opening.py'
    spec = importlib.util.spec_from_file_location('tested_pavement_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local_path=args.local_root/'engine/pavement_obstructions.py'
    local_spec=importlib.util.spec_from_file_location('tested_obstruction_module',local_path)
    local_module=importlib.util.module_from_spec(local_spec)
    local_spec.loader.exec_module(local_module)
    args.out.mkdir(parents=True, exist_ok=True)
    named = next(j for j in suite.jobs(['fresh']) if j[1]['id']==COLONNADE)
    records, panels = counterfactuals(module.inspect_openings, local_module.window_obstructions, named)
    jobs = [j for j in suite.jobs(['new','shelter']) if j[1]['family'] in ('switchback','loadingdock','platformshelter')]
    native = [measured(module.inspect_openings, j) for j in jobs]
    save(args.out/'controls.json', dict(default_revision='a6e0195', candidate_sha256=digest(path), local_candidate_sha256=digest(local_path),
         opening_sha256=named[1]['files']['rgb.png'], controlled=records, native=native))
    write_comparison(panels, args.out/'platform-comparison.jpg')
    for row in records:
        print(row['control'],row['input_pixels'],row['candidate_reached_phase_check'],row['suppressed'])
    print('Native tops', sum(r['block_faces'] for r in native),'visible pixels',
          sum(r['platform_pixels'] for r in native),'covered',sum(r['platform_pixels_in_input_masks'] for r in native),
          'removed',sum(r['removed_platform_pixels'] for r in native))


if __name__ == '__main__':
    main()
