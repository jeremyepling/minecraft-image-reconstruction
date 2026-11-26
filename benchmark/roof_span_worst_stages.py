"""Read and display the recorded surface masks and count charts on their original pixels."""
import numpy as np
from PIL import Image, ImageDraw

from benchmark import framed_depth_eval as framed
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.surface_pipeline_sheet import _picture, _straight_photo, MAT
from benchmark.texture_camera_eval import checked, read
from engine import atlas_count, surface_perimeter
from engine.atlas_phase import register_phase
from engine.geometry import project, transform
from engine.surface_sample import sample_surface
from engine.surface_tiles import visible_tiles
from training.visible_labels import visible_labels


def read_view(run, split, frame, prior, row):
    key, hashes = frame['id'], frame['files']
    image, parts, counts, camera = framed.photo(split, frame, prior)
    truth, _, registration, input_hashes = framed.references(split, frame, framed.NATIVE_INPUTS.get(split),
        framed.TRAINING_DATA, framed.DEPTH, dict(native=framed.native.BUILDS,
        old=framed.transfer.BUILDS, fresh=framed.transfer.BUILDS))
    assert row['input_hashes'] == input_hashes and row['reference_hashes'] == hashes
    pack = decoded_pack(checked(run / split / key / 'reconstruction.zip', row['pack_sha256']))
    raw = registered(set(pack), registration)
    local = read(run / split / key / 'cells.json')
    assert raw == set(map(tuple,local['world'])) and set(pack) == set(map(tuple,local['local']))
    scored = framed.spacing.shift(raw, row['translation'])
    assert len(scored & truth) == row['relative_geometry']['correct']
    if split in ('old','fresh','native'):
        rgb = framed.native.rgb_path(frame) if split == 'native' else framed.transfer.rgb_path(frame)
        name = key if split == 'native' else frame['source_id']
        pose = read(checked(rgb.with_suffix('.json'), hashes['camera.json']))
        states = read(checked(rgb.with_name(name+'-blocks.json'), hashes['world-readback.json']))
        states = {tuple(b[k] for k in 'xyz'):b for b in states}
        labels, ids, _ = visible_labels(pose, states, (0,0,0))
    else:
        base = framed.TRAINING_DATA / 'data' / split / frame['directory']
        pose = read(checked(base/'camera.json', hashes['camera.json']))
        labels = read(checked(base/'labels.json', hashes['labels.json']))
        with np.load(checked(base/'labels.npz', hashes['labels.npz'])) as labels_array:
            ids = labels_array['face_id'].copy()
    assert truth == {tuple(b['xyz_world']) for b in labels['blocks']}
    return image, parts, counts['surfaces'], camera, pose, labels, ids, truth, raw, scored, sorted(set(pack.values()))


def surface_evidence(photo, parts, counts, labels, ids, missed, atlas):
    lut = np.zeros(len(labels['faces'])+1, dtype=int)
    missed_ids = []
    for face in labels['faces']:
        lut[face['id']] = face['surface_id']
        if tuple(face['xyz_world']) in missed:
            missed_ids.append(face['id'])
    truth_masks = lut[ids]
    size = np.bincount(truth_masks.ravel())
    missing_pixels = np.isin(ids, missed_ids)
    by_id = {c['id']: c for c in counts}
    results, pairs = [], []
    for part in parts:
        identifier = part['id']
        record = by_id[identifier]
        alpha = surface_perimeter._native_alpha(part['support'], photo.size)
        if alpha is None:
            continue
        mask, box = alpha
        left,top,right,bottom = box
        owned = np.asarray(mask).astype(bool)
        areas = np.bincount(truth_masks[top:bottom,left:right][owned],minlength=len(size))
        ious = areas[1:]/(owned.sum()+size[1:]-areas[1:])
        best = int(np.argmax(ious))+1
        pairs.extend((float(ious[i]),identifier,int(i+1)) for i in np.flatnonzero(ious>=.5))
        # Rebuild the exact chosen count chart, including any texture-guided rectification.
        sample = sample_surface(photo, dict(part, quad=record['quad'])) if record.get('quad') else None
        if sample is not None:
            assert np.allclose(sample['quad'],record['quad'])
        tiles = list(visible_tiles(sample,record)) if sample is not None else []
        replay = register_phase(sample,record,atlas)
        replay_tiles = tiles if replay['phase'] == record['phase'] else list(visible_tiles(sample,replay))
        results.append(dict(part=part,record=record,mask=mask,box=box,sample=sample,tiles=tiles,
            replay=replay,replay_tiles=replay_tiles,
            missing=int(missing_pixels[top:bottom,left:right][owned].sum()),area=int(owned.sum()),
            matched_surface=best,best_iou=float(ious[best-1])))
    paired_preds, paired_truth = set(), set()
    for _,identifier,surface in sorted(pairs,reverse=True):
        if identifier not in paired_preds and surface not in paired_truth:
            paired_preds.add(identifier)
            paired_truth.add(surface)
    def measured(result):
        return all(v is not None for v in result['record']['dimensions']['blocks'])
    def located(result):
        return all(v is not None for v in result['record']['phase'])
    totals = dict(predicted=len(parts),visible=len(size)-1,matched_at_iou_half=len(paired_truth),
                  charts=sum(r['sample'] is not None for r in results),
                  two_dimension_counts=sum(measured(r) for r in results),
                  two_grid_origins=sum(located(r) for r in results),
                  complete_tiles=sum(len(r['tiles']) for r in results),
                  atlas_replay_tiles=sum(len(r['replay_tiles']) for r in results))
    def display_priority(r):
        c = r['record']
        return (bool(r['tiles']),bool(r['replay_tiles']),r['sample'] is not None,
                any(v is not None for v in c['phase']),any(v is not None for v in c['counts']),
                r['missing'],r['area'],r['part']['id'])
    selected = max(results, key=display_priority)
    return results, selected, totals


def mark_source_grid(photo, results, selected):
    marked = photo.copy()
    grids = Image.new('L',photo.size,0)
    selected_grid = Image.new('L',photo.size,0)
    for r in results:
        sample = r['sample']
        if sample is None:
            continue
        result = r['record']
        quad = np.asarray(sample['quad'])
        homography = transform(quad)
        x0,y0,x1,y1 = r['box']
        lines = Image.new('L',(x1-x0,y1-y0))
        pen = ImageDraw.Draw(lines)
        for axis,(count,phase) in enumerate(zip(result['counts'],result['phase'])):
            if count is None or phase is None or count<=0:
                continue
            for uv in np.arange(phase%1.,count+1e-8,1.)/count:
                ends = np.array([[uv,0],[uv,1]]) if axis==0 else np.array([[0,uv],[1,uv]])
                pixels = project(homography,ends)-[x0,y0]
                pen.line(list(map(tuple,pixels)), fill=255, width=3 if r is selected else 2)
        crop = np.minimum(np.asarray(lines),np.asarray(r['mask']))
        grids.paste(255,r['box'],Image.fromarray(crop))
        if r is selected:
            selected_grid.paste(255,r['box'],Image.fromarray(crop))
    marked.paste('#efa335',(0,0,photo.width,photo.height),grids)
    marked.paste('#fff36a',(0,0,photo.width,photo.height),selected_grid)
    surface_perimeter.draw_outline(marked,selected['part'],'#40ffff',2)
    return marked


def material_pair(selected, atlas):
    sample,record = selected['sample'],selected['record']
    if sample is None:
        return None,None,None,dict(reason='No measured chart')
    measured = next(iter(selected['tiles']),None)
    clean, grid, _ = _straight_photo(sample,record,measured['cell'] if measured else None)
    tile = measured if measured else next(iter(selected['replay_tiles']),None)
    if tile is None:
        return clean,grid,None,dict(reason='Atlas offset replay found no complete supported tile')
    lookup = atlas_count.rank_tile(tile,atlas,version='26.3')
    img = Image.new('RGB',(408,196),MAT)
    _picture(img,Image.fromarray(tile['rgb']),0,0,196,196,nearest=True)
    texture = lookup.get('texture')
    if texture:
        _picture(img,Image.fromarray(atlas.tiles[atlas.labels.index(texture)]),212,0,196,196,nearest=True)
    return clean,grid,img,dict(cell=tile['cell'],quad=tile['quad'],
                              origin_source='frozen chart' if measured else 'independent atlas offset replay',
                              phase=selected['replay']['phase'],
                              measured_block_id=lookup.get('block_id'),texture=texture,
                              ranked=lookup.get('ranked_ids',[]))
