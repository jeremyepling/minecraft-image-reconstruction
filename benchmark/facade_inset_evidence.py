"""Grade the seven photographed Facade recesses using visible source faces only."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import best_translation
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from benchmark.visible_error_budget import projection
from engine.pillar_seams import bounds, native_mask

KEY = 'heldout-facade-oblique-right'
FROZEN = Path('../visible-error-budget/runs/primary-production')
ORDINARY = Path('../strip-orientation-audit/runs/final-facade')
OUT = Path('benchmark/reviews/facade-insets-01')
# Physical recesses in reading order. Surface IDs come exclusively from grading labels.
OPENINGS = {
    'A': ('Small upper left', 4, 5, 6),
    'B': ('Upper middle', 12, 13, 14),
    'C': ('Upper right', 17, 18, 19),
    'D': ('Middle left', 7, 8, 9),
    'E': ('Ground left', 1, None, 3),
    'F': ('Ground middle', 10, None, 11),
    'G': ('Ground right', 15, None, 16),
}
PRIMARY = ('D', 'B', 'C')
COLORS = {'return': (240,164,59), 'sill': (74,203,225), 'back': (87,211,148)}
NAMES = {'return': '+X return', 'sill': '+Y sill', 'back': '+Z back'}


def frozen():
    row = next(r for r in read(checked(FROZEN/'cases.json', read(FROZEN/'summary.json')['cases_sha256'])) if r['id'] == KEY)
    split, frame, prior = next(x for x in framed.cases((row['split'],)) if x[1]['id'] == KEY)
    photo, _, _, camera, pose, labels, ids, truth, *_ = read_view(FROZEN, split, frame, prior, row)
    return photo, camera, pose, labels, ids, truth, row


def projected(pack, camera, pose, truth, ids):
    local = sorted(decoded_pack(pack))
    p = np.asarray(camera['P'], dtype=float)
    transform = registration(p/np.linalg.norm(p[2,:3]), pose)
    shift, correct = best_translation(registered(local, transform), truth)
    owner, axis, axes, world = projection(camera, pose, local, shift, ids.shape)
    cells = set(map(tuple, world.tolist()))
    assert len(cells & truth) == correct
    return dict(owner=owner, axis=axes[axis], world=world, cells=cells,
                metrics=dict(translation=list(shift), correct=correct, missed=len(truth-cells), extras=len(cells-truth)))


def masked_first_hits(target, which, xyz, normal, geo):
    owner = geo['owner'][target]
    valid = owner >= 0
    axis = int(np.argmax(np.abs(normal)))
    # Empty pixels carry no first-hit owner. Only compare actual owners.
    coords = geo['world'][owner[valid]]
    reference = xyz[which[target][valid]]
    delta = (coords[:,axis]-reference[:,axis])*normal[axis]
    return dict(uncovered=int((~valid).sum()), correct_cell=int((coords==reference).all(1).sum()),
                same_normal_axis=int((geo['axis'][target][valid]==axis).sum()),
                first_hit_in_front=int((delta>0).sum()), first_hit_on_plane=int((delta==0).sum()),
                first_hit_behind=int((delta<0).sum()))


def grade(surfaces, counts, pack):
    photo, camera, pose, labels, ids, truth, row = frozen()
    faces = labels['faces']
    blocks = {b['id']:b for b in labels['blocks']}
    sid = np.zeros(len(faces)+1, dtype=int)
    xyz = np.zeros((len(faces)+1,3),dtype=int)
    for f in faces:
        sid[f['id']], xyz[f['id']] = f['surface_id'], f['xyz_world']
    source = sid[ids]
    parts = read(surfaces)['planes']
    masks = {p['id']:native_mask(p, photo.size) for p in parts}
    charts = {c['id']:c for c in read(counts)['surfaces']} if counts else {}
    def iou(m, target):
        intersection = int((m & target).sum())
        return intersection / int((m | target).sum())
    host = max(masks, key=lambda i:iou(masks[i], source==2)) if masks else None
    hostmask = masks[host] if host else np.zeros_like(source, bool)
    geo = projected(pack,camera,pose,truth,ids) if pack else None
    result = dict(frame=KEY, split=row['split'], rgb_sha256=row['rgb_sha256'], native_size=list(photo.size),
                  visible_reference=row['reference_hashes'], production_case_manifest_sha256=digest(FROZEN/'cases.json'),
                  surface_sha256=digest(surfaces), host_mask=host, openings={})
    target_host = source==2
    result['facade_mask'] = dict(correct_px=int((hostmask & target_host).sum()),
        missed_px=int((~hostmask & target_host).sum()), extra_px=int((hostmask & ~target_host).sum()),
        background_px=int((hostmask & (ids==0)).sum()), iou=iou(hostmask,target_host))
    manifest = surfaces.with_name('build-manifest.json')
    if manifest.exists():
        assert read(manifest)['image_sha256']==row['rgb_sha256']
    if counts:
        result['counts_sha256']=digest(counts)
    if geo:
        result.update(pack_sha256=digest(pack), strict_geometry=geo['metrics'])
    for letter, (name, side, sill, back) in OPENINGS.items():
        sids = [s for s in (side,sill,back) if s is not None]
        opening = np.isin(source, sids)
        out = dict(name=name, bounds_xyxy_exclusive=list(bounds(opening)), area_px=int(opening.sum()),
                   host_px=int((hostmask & opening).sum()), faces={})
        for role, group in zip(('return','sill','back'), (side,sill,back)):
            if group is None:
                continue
            target = source==group
            group_faces = [f for f in faces if f['surface_id']==group]
            normal = group_faces[0]['normal_world']
            cells = {tuple(f['xyz_world']) for f in group_faces}
            area = int(target.sum())
            best = max(masks, key=lambda i:iou(masks[i],target)) if masks else None
            chart = charts.get(best)
            face = dict(surface=group, normal=normal, coordinates=sorted(cells),
                        visible_face_ids=[f['id'] for f in group_faces], bounds_xyxy_exclusive=list(bounds(target)),
                        pixels=area, host_px=int((hostmask & target).sum()), best_mask=best,
                        best_mask_iou=iou(masks[best],target) if best is not None else 0.,
                        materials=dict(Counter(blocks[f['block']]['block_id'] for f in group_faces)))
            if chart:
                face.update(best_mask_count=chart['dimensions']['blocks'], best_mask_phase=chart['phase'])
            if geo:
                face.update(matched_positions=sorted(cells & geo['cells']), missed_positions=sorted(cells - geo['cells']),
                            first_hits=masked_first_hits(target,ids,xyz,normal,geo))
            out['faces'][role] = face
        result['openings'][letter] = out
    patch = [f for f in faces if f['surface_id']==2 and blocks[f['block']]['block_id']=='minecraft:polished_deepslate']
    patch_mask=np.isin(ids,[f['id'] for f in patch])
    patch_cells={tuple(f['xyz_world']) for f in patch}
    result['flat_material_control']=dict(surface=2,material='minecraft:polished_deepslate',coordinates=sorted(patch_cells),
        pixels=int(patch_mask.sum()),host_px=int((hostmask & patch_mask).sum()),bounds_xyxy_exclusive=list(bounds(patch_mask)))
    if geo:
        result['flat_material_control']['matched_positions']=sorted(patch_cells & geo['cells'])
    image_data=(photo,source,hostmask,masks)
    return result,image_data


def overlay(rgb,targets,alpha=.42):
    arr = np.array(rgb)
    for mask, color in targets:
        inside = np.pad(mask,1)
        core = inside[1:-1,:-2] & inside[1:-1,2:] & inside[:-2,1:-1] & inside[2:,1:-1]
        arr[mask] = ((1-alpha)*arr[mask]+alpha*np.asarray(color)).astype(np.uint8)
        arr[mask & ~core]=color
    return Image.fromarray(arr)


def figures(q, image_data, folder):
    photo, source, hostmask, masks = image_data
    # Keep the complete photo in its original camera frame. Letter badges sit outside openings.
    whole=photo.copy()
    pen=ImageDraw.Draw(whole)
    for tag,r in q['openings'].items():
        x,y,b,t=r['bounds_xyxy_exclusive']
        color='#ffee8a' if tag in PRIMARY else 'white'
        pen.rectangle((x-2,y-2,b+1,t+1),outline=color,width=2)
        pen.rounded_rectangle((x-24,y-23,x-1,y-1),3,fill='#192830',outline=color)
        pen.text((x-20,y-24),tag,font=font(18),fill=color)
    # Label the flush masonry patch as a control in the same photograph.
    x,y,b,t=q['flat_material_control']['bounds_xyxy_exclusive']
    pen.rectangle((x-1,y-1,b+1,t+1),outline='#bba0f0',width=2)
    pen.text((x+4,y+2),'H',font=font(16),fill='white',stroke_width=2,stroke_fill='#362c43')
    whole.save(folder/'facade-marked.jpg',quality=94)
    sheet=Image.new('RGB',(1080,805),'#142832'); d=ImageDraw.Draw(sheet)
    def text(x,y,s,size=17,color='white'):
        d.text((x,y),s,font=font(size),fill=color)
    text(18,12,'Three recesses whose visible backs blend into the facade mask',24)
    text(18,48,'Same photograph in each column. All three recessed backs and their surrounding wall are tuff.',18)
    for x,h in [(20,'Photograph'),(277,'Current facade mask'),(534,'Recorded visible surfaces'),(802,'Visible back blocks')]:
        text(x,95,h,19)
    for j,tag in enumerate(PRIMARY):
        r=q['openings'][tag]; y=133+202*j
        x0,y0,x1,y1=r['bounds_xyxy_exclusive']
        cut=(x0-9,y0-9,x1+9,y1+9)
        parts=[(source==f['surface'],COLORS[role]) for role,f in r['faces'].items()]
        panels=[photo, overlay(photo,[(hostmask,(239,116,92))]), overlay(photo,parts)]
        for i,panel in enumerate(panels):
            small=ImageOps.contain(panel.crop(cut),(239,139),Image.Resampling.NEAREST)
            sheet.paste(small,(19+i*258+(240-small.width)//2,y+(140-small.height)//2))
        back=r['faces']['back'];m=len(back['matched_positions']);n=len(back['coordinates'])
        text(21,y+143,tag+'   '+r['name'],19)
        text(278,y+145,f"{back['host_px']/back['pixels']:.1%} of back under host",15)
        text(802,y+31,f'{m} / {n} placed',24)
        text(802,y+66,f"Depth {abs(back['coordinates'][0][2])} block"+('s' if abs(back['coordinates'][0][2])!=1 else ''),18)
        text(802,y+95,'behind front wall',16)
    for i,name in enumerate(('return','sill','back')):
        x=26+i*232;d.rectangle((x,756,x+18,774),fill=COLORS[name]);text(x+26,754,NAMES[name],17)
    text(747,754,'Salmon = current front wall',17)
    sheet.save(folder/'inset-detail.jpg',quality=94)


def comparison_sheet(q, image_data, folder):
    before, (_, _, original_host, _) = grade(ORDINARY/'surfaces.json',None,None)
    photo, source, new_host, masks = image_data
    sheet=Image.new('RGB',(1410,1088),'#142832')
    pen=ImageDraw.Draw(sheet)
    def write(x,y,caption,px=19,color='white'):
        pen.text((x,y),caption,font=font(px),fill=color)
    write(24,12,'Facade  ·  source, original Surface Finder, proposed split and recorded labels',28)
    write(24,55,'Same 1440 × 900 photograph throughout. Colored overlays below mark the pixels assigned by each method.',19)
    thumb=ImageOps.contain(photo,(680,405),Image.Resampling.LANCZOS)
    left,top=18,94
    sheet.paste(thumb,(left,top))
    for tag,r in q['openings'].items():
        x,y,x1,y1=r['bounds_xyxy_exclusive']
        a=left+x*thumb.width/photo.width; b=top+y*thumb.height/photo.height
        c=left+x1*thumb.width/photo.width; d=top+y1*thumb.height/photo.height
        color='#fbe28f' if tag in PRIMARY else '#ffffff'
        pen.rectangle((a-1,b-1,c+1,d+1),outline=color,width=2)
        pen.rectangle((a-17,b-19,a-1,b-1),fill='#243946')
        write(a-16,b-20,tag,15,color)
    write(707,129,'Seven visible recesses',25)
    write(707,178,'A  small upper left      B  upper middle      C  upper right',19)
    write(707,214,'D  middle left               E  ground left          F  ground middle',19)
    write(707,250,'G  ground right',19)
    write(707,313,'D, B and C have tuff on both the facade and the back.',20)
    write(707,352,'A left return and a sill mark each change in depth.',20)
    write(707,391,'The close-ups use matching crops and the same scale per row.',17,'#d2dfdf')
    columns=[(20,'Photograph'),(317,'Original mask'),(612,'Proposed split'),(907,'Recorded faces')]
    for x,title in columns:write(x,517,title,21)
    write(1204,517,'Back-mask IoU',19)
    for j,tag in enumerate(PRIMARY):
        r=q['openings'][tag]; bb=r['faces']['back']; prev=before['openings'][tag]['faces']['back']
        x,y,x1,y1=r['bounds_xyxy_exclusive']; bx=(x-8,y-8,x1+8,y1+8)
        pred=[(new_host,(235,112,88))]
        if bb['best_mask_iou']>=.5:
            pred.append((masks[bb['best_mask']],COLORS['back']))
        reference=[(source==f['surface'],COLORS[role]) for role,f in r['faces'].items()]
        panels=[photo,overlay(photo,[(original_host,(235,112,88))]),overlay(photo,pred),overlay(photo,reference)]
        yy=560+j*150
        for i,im in enumerate(panels):
            crop=ImageOps.contain(im.crop(bx),(276,134),Image.Resampling.NEAREST)
            sheet.paste(crop,(columns[i][0]+(276-crop.width)//2,yy+(134-crop.height)//2))
        write(1204,yy+20,tag+'   '+r['name'],18)
        write(1204,yy+56,f"{prev['best_mask_iou']:.2f} to {bb['best_mask_iou']:.2f}",22)
    write(24,1034,'Salmon  ·  front wall',18,'#ffab99')
    write(294,1034,'Green  ·  back (+Z)',18,'#87e6ad')
    write(605,1034,'Orange  ·  return (+X)',18,'#f6c268')
    write(922,1034,'Blue  ·  sill (+Y)',18,'#8be8f5')
    sheet.save(folder/'before-after.jpg',quality=94)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surfaces',type=Path,help='Candidate surfaces.json from this exact photograph and camera')
    parser.add_argument('--counts',type=Path)
    parser.add_argument('--pack',type=Path,help='Candidate reconstruction.zip using the same frozen estimated camera')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--comparison',action='store_true',help='Draw the candidate and baseline masks on the original pixels')
    args=parser.parse_args()
    baseline=args.surfaces is None
    surfaces=args.surfaces or ORDINARY/'surfaces.json'
    counts=args.counts or (ORDINARY/'counts.json' if baseline else None)
    pack=args.pack or (ORDINARY/'reconstruction.zip' if baseline else None)
    q,image_data=grade(surfaces,counts,pack)
    if baseline:
        assert q['surface_sha256']==read(Path('benchmark/reviews/strip-orientation-01/wall-extent.json'))['surfaces_sha256']
        assert decoded_pack(pack)==decoded_pack(FROZEN/q['split']/KEY/'reconstruction.zip')
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'openings.json',q)
    if baseline:
        figures(q,image_data,args.out)
    if args.comparison:
        comparison_sheet(q,image_data,args.out)


if __name__=='__main__':
    main()
