"""Verify ordinary exports against the paired replay and illustrate the two affected views."""
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import facade_inset_round as suite
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.pavement_opening_eval import priority, visible_faces
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import read, save, digest
from benchmark.texture_camera_sheet import font
from benchmark.thin_face_placement_run import COHORTS, ref_inputs
from engine.pillar_seams import native_mask, orientation
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE

REVIEW = suite.ROOT/'benchmark/reviews'
OUT = REVIEW/'pavement-local-01'


def rate(counts):
    return 2*counts['correct']/(2*counts['correct'] + counts['missed'] + counts['extras'])


def overlay(image, mask, tint):
    rgb = np.asarray(image).copy()
    rgb[mask] = .35*rgb[mask] + .65*np.asarray(tint)
    return Image.fromarray(rgb)


def sample_grid(image, camera, opening):
    """Locate integer ground-grid crossings between adjacent source pixels."""
    projection, _ = orientation(camera, image.size)
    inv = np.linalg.inv(projection[:, :3])
    eye = -inv @ projection[:, 3]
    y, x = np.where(opening)
    r = np.stack((x+.5, y+.5, np.ones(len(x))), axis=1) @ inv.T
    xyz = eye + r * (-eye[1]/r[:, 1:2])
    cell = np.zeros((*opening.shape, 2), np.int32)
    cell[y, x] = np.floor(xyz[:, [0,2]]).astype(int)
    grid = np.zeros_like(opening)
    grid[:, 1:] |= opening[:, 1:] & opening[:, :-1] & (cell[:, 1:] != cell[:, :-1]).any(2)
    grid[1:, :] |= opening[1:, :] & opening[:-1, :] & (cell[1:, :] != cell[:-1, :]).any(2)
    return overlay(image, grid, (248, 165, 24))


def marked_spill(image, camera, cells, transform, truth, opening):
    ordered = sorted(cells)
    p, _ = orientation(camera, image.size)
    eye = -np.linalg.solve(p[:, :3], p[:, 3])
    p[:2] *= SAMPLE_STRIDE
    owner = _frontmost_cells(ordered, p, eye, image.size[::-1])
    extras = [j for j, c in enumerate(ordered) if not registered([c],transform) <= truth]
    return overlay(image, np.isin(owner, extras), (241,77,79)), int(np.sum(opening & (owner >= 0)))


def render_column(board, column, view, image, mask, camera, before, after, transform, truth):
    x = 207 + column*490
    ink='#153239'
    pen = ImageDraw.Draw(board)
    pen.text((x, 72), view.capitalize()+' view', fill=ink, font=font(23))
    bounds = Image.fromarray(mask).getbbox()
    l,t,r,b = bounds
    box = (max(0,l-38),max(0,t-30),min(image.width,r+38),min(image.height,b+30))
    full=image.copy()
    ImageDraw.Draw(full).rectangle(box, outline='#f39f0b',width=4)
    previous, before_hits = marked_spill(image, camera, before, transform, truth, mask)
    updated, after_hits = marked_spill(image, camera, after, transform, truth, mask)
    stages=[full,sample_grid(image,camera,mask).crop(box),previous.crop(box),updated.crop(box)]
    for j, frame in enumerate(stages):
        tile=ImageOps.contain(frame,(468,218),Image.Resampling.LANCZOS if j==0 else Image.Resampling.NEAREST)
        board.paste(tile,(x+(468-tile.width)//2,109+j*232+(218-tile.height)//2))
    return dict(opening_pixels=int(mask.sum()), blocked_before=before_hits, blocked_after=after_hits)


def main():
    rows=read(OUT/'cases.json')['cases']
    full={r['id']:r for r in read(REVIEW/'pavement-opening-round-01/cases.json')}
    totals={}
    for name,splits in COHORTS.items():
        records=[r['baseline'] for r in full.values() if r['split'] in splits]
        totals[name]={}
        for metric in ('strict','facade','registered'):
            pool=Counter()
            for r in records:
                pool.update(r[metric])
            totals[name][metric]=dict(pool)
    changed=[]
    native_platforms=[]
    candidates={r['id']:r for r in rows}
    assert len(rows)==len(full)==204
    board=Image.new('RGB',(1187,1041),'#faf8f0')
    pen=ImageDraw.Draw(board)
    pen.text((15,8),'Colonnade 01   |   Pavement through the opening',font=font(32),fill='#153239')
    pen.text((16,52),'Orange lines follow the projected ground grid. Red marks misplaced blocks visible from the source camera.',font=font(18),fill='#385456')
    for j,title in enumerate(('Source image','Pavement grid','Previous placement','Current placement')):
        pen.text((12,135+j*232),title,font=font(19),fill='#153239')
    zpath=OUT/'default-exports.zip'
    with ZipFile(zpath,'w',ZIP_DEFLATED) as archive:
        for job in suite.jobs([*suite.framed.SIZES,'shelter']):
            sp,f,_,_=job
            key=f['id']
            row=candidates[key]
            old=full[key]['baseline']
            assert row['source_hash']==old['rgb_sha256'] and row['pack_hash']==old['pack_sha256']
            if key in ('holdout-loadingdock-timber-high-right','holdout-platformshelter-copper-high-right'):
                path = suite.ROOT/'runs'/('pavement-cli-'+f['family'])
                build = read(path/'build-manifest.json')
                assert build['image_sha256'] == row['source_hash'] and build['through'] == 'placement'
                _, expected = baseline_pack(job)
                assert decoded_pack(path/'reconstruction.zip') == dict.fromkeys(expected,'minecraft:smooth_stone')
                assert not row['removed'] and not build['placement'].get('pavement_openings')
                native_platforms.append(dict(id=key, source_hash=row['source_hash'], blocks=len(expected),
                                             ordinary_pack_equal=True, pack_sha256=digest(path/'reconstruction.zip')))
            if not row['removed']:
                continue
            view=key.rsplit('-',1)[-1]
            path=suite.ROOT/'runs'/('pavement-cli-'+view)
            im,parts,_,cam=suite.ordinary_inputs(*job)
            _,before=baseline_pack(job)
            after=before-set(map(tuple,row['removed']))
            build=read(path/'build-manifest.json')
            assert build['image_sha256']==row['source_hash'] and build['through']=='placement'
            assert {tuple(c) for c in build['placement']['pavement_openings']['removed']}==before-after
            assert decoded_pack(path/'reconstruction.zip')==dict.fromkeys(after,'minecraft:smooth_stone')
            assert {tuple(b[k] for k in 'xyz') for b in read(path/'placements.json')['blocks']}==after
            archive.write(path/'reconstruction.zip',key+'/reconstruction.zip')
            truth,occupied,transform,_=ref_inputs(job)
            real=registered(after,transform)
            scores=placement_scores(real,registered(before,transform),dict.fromkeys(truth),occupied)
            labels,ids=visible_faces(sp,f)
            opt=horizontal_slivers(labels,ids)
            shifted=suite.framed.spacing.shift(real,scores['translation'])
            assert scores['relative_geometry']==row['relative']
            new=dict(strict=row['relative'],registered=row['registered'],facade=priority(shifted,truth,opt))
            group=next(k for k,v in COHORTS.items() if sp in v)
            for m in new:
                for field in old[m]:
                    totals[group][m][field]+=new[m][field]-old[m][field]
            levels=[r for r in row['proposed'] if r['reject']]
            keys={r['id'] for r in levels}
            mask=np.logical_or.reduce([native_mask(p,im.size) for p in parts if p['id'] in keys])
            assert keys==set(build['placement']['pavement_openings']['surfaces'])
            rays = render_column(board, len(changed), view, im, mask, cam, before, after, transform, truth)
            changed.append(dict(id=key,source_hash=row['source_hash'],previous_pack_sha256=row['pack_hash'],
                pack_sha256=digest(path/'reconstruction.zip'), ordinary_placements_equal=True,
                removed=sorted(before-after), baseline={m:old[m] for m in new},
                result={m:{k:v[k] for k in old[m]} for m,v in new.items()}, sightlines=rays, evidence=levels))
    for group in totals.values():
        for record in group.values():
            record['f1']=rate(record)
    retired=read(REVIEW/'facade-local-placement-01/summary.json')['historical_texture_controls']['visible_targets']
    hist=totals['fixed']['strict'].copy()
    hist['missed']+=retired
    assert len(changed)==2 and len(native_platforms)==2
    board.save(OUT/'openings.jpg',quality=94,subsampling=0)
    save(OUT/'summary.json',dict(baseline='a6e0195',frames=len(rows),changed=changed,cohorts=totals,
        ordinary_platforms=native_platforms, cases_sha256=digest(OUT/'cases.json'),
        historical_112_strict_f1=rate(hist),archive_sha256=digest(zpath)))
    print(totals['fixed'])


if __name__=='__main__':
    main()
