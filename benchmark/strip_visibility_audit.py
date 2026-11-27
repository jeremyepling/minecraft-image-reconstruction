"""Rescore frozen exports by native visible patch size and locate misses and extras near thin faces."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as frames
from benchmark.facade_visibility import rates, square_support
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from benchmark.visible_error_budget import projection, stage_evidence, tally
from engine.surface_model import intrinsic
from engine.surface_perimeter import _native_alpha
from training.visible_labels import visible_labels


SIDES = (2, 4, 8, 16)
GROUPS = ('top_only', 'underside_only', 'any_wall')


def view(run, split, frame, inputs, row):
    if split != 'extra':
        image, parts, counts, camera, pose, labels, ids, truth, _, placed, _ = read_view(run, split, frame, inputs, row)
    else:
        image, parts, measured, camera = frames.photo(split, frame, inputs)
        counts = measured['surfaces']
        prefix = frames.transfer.BUILDS/frame['directory']/'captures'/frame['source_id']
        hashes = frame['files']
        pose = read(checked(prefix.with_suffix('.json'), hashes['camera.json']))
        states = read(checked(prefix.with_name(prefix.name+'-blocks.json'), hashes['world-readback.json']))
        labels, ids, _ = visible_labels(pose, {tuple(b[k] for k in 'xyz'):b for b in states}, (0,0,0))
        truth, _, transform, input_hashes = frames.references('old',frame,frames.NATIVE_INPUTS[split],
            frames.TRAINING_DATA,frames.DEPTH,dict(old=frames.transfer.BUILDS))
        assert row['input_hashes'] == input_hashes and row['reference_hashes'] == hashes
        cells = set(decoded_pack(checked(run/split/frame['id']/'reconstruction.zip',row['pack_sha256'])))
        placed = frames.spacing.shift(registered(cells, transform),row['translation'])
    assert truth == {tuple(b['xyz_world']) for b in labels['blocks']}
    assert row['rgb_sha256'] == inputs['rgb_sha256'] == frame['files']['rgb.png']
    return image, parts, counts, camera, pose, labels, ids, truth, placed


def figure(path, image, ids, owners, hit, top_only, sliver, locally_thin):
    fg = ids > 0
    color = np.array(image)
    colors = np.array([[31,208,138],[255,176,49],[230,38,146],[35,160,255]])
    category = np.where(hit,0,np.where(sliver,2,np.where(locally_thin,3,1)))
    color[fg] = .30*color[fg] + .70*colors[category[owners[ids[fg]]]]
    detail = fg & top_only[owners[ids]] & ~hit[owners[ids]]
    yy,xx = np.nonzero(detail)
    ys,xs = np.nonzero(fg)
    building = (int(xs.min())-8,int(ys.min())-8,int(xs.max())+9,int(ys.max())+9)
    # Cropping only changes presentation; all measurements above use native pixels.
    crop = (int(xx.min())-3,int(yy.min())-5,int(xx.max())+4,int(yy.max())+6)
    page = Image.new('RGB',(1180,716),'#152b35')
    ink = ImageDraw.Draw(page)
    ink.text((16,8),'Colonnade 01    |    Visible geometry in the original screenshot',font=font(25),fill='white')
    pieces = ((image,(16,62),(400,250)),(image.crop(building),(448,62),(704,250)),
              (Image.fromarray(color).crop(crop),(16,368),(1130,115)))
    for pic,(x,y),size in pieces:
        scale=min(size[0]/pic.width,size[1]/pic.height)
        tile=pic.resize((int(pic.width*scale),int(pic.height*scale)),Image.Resampling.NEAREST)
        page.paste(tile,(x,y))
    ink.text((16,314),'Complete input',font=font(17),fill='white')
    ink.text((448,314),'Building detail',font=font(17),fill='white')
    yb,xb=np.nonzero(detail & (locally_thin&~sliver)[owners[ids]])
    if len(xb):
        i=np.argmin(abs(xb-np.median(xb)))
        x,y=int(xb[i]),int(yb[i])
        bounds=(x-43,y-9,x+44,y+10)
        for offset,source in ((16,image),(606,Image.fromarray(color))):
            patch=source.crop(bounds).resize((522,114),Image.Resampling.NEAREST)
            page.paste(patch,(offset,486))
        ink.text((16,604),'Same strip at 6×, source pixels',font=font(16),fill='white')
        ink.text((606,604),'Same 6× crop, visible-block grading',font=font(16),fill='white')
    pink = np.count_nonzero(sliver & ~hit)
    blue = np.count_nonzero(locally_thin & ~sliver & ~hit)
    ink.text((16,634),f'Pink   {pink} misses on top surfaces without an 8 × 8 patch',font=font(18),fill='#ff93d2')
    ink.text((16,662),f'Blue   {blue} additional misses with a locally thin top face on a wider roof surface',font=font(18),fill='#7dc4ff')
    ink.text((16,690),'Green   correct placements     Gold   missed wall blocks',font=font(16),fill='white')
    page.save(path,quality=92)


def block_footprints(labels,ids):
    """Return orientation bits and the largest tested native squares on each block and its surfaces."""
    faces,blocks=labels['faces'],labels['blocks']
    assert [b['id'] for b in blocks] == list(range(1,len(blocks)+1))
    assert [f['id'] for f in faces] == list(range(1,len(faces)+1))
    owners=np.array([0]+[f['block'] for f in faces])
    surface=np.array([0]+[f['surface_id'] for f in faces])
    axes=np.array([-1]+[np.argmax(np.abs(f['normal_world'])) for f in faces])
    n=len(blocks)+1
    sizes=np.bincount(owners[ids.ravel()],minlength=n)
    assert (sizes[1:]>0).all()
    fsize=square_support(ids,len(faces))
    ssize=square_support(surface[ids],int(surface.max()))
    min_face_needed=np.zeros(n,int)
    min_surf_needed=np.zeros(n,int)
    # A coordinate is optional only when ALL its photographed evidence is thin.
    np.maximum.at(min_face_needed,owners[1:],fsize[1:])
    np.maximum.at(min_surf_needed,owners[1:],ssize[surface[1:]])
    bits=np.zeros(n,int)
    np.bitwise_or.at(bits,owners[1:],np.array([1 if f['normal_world'][1]>0 else
        2 if f['normal_world'][1]<0 else 4 for f in faces]))
    return owners, axes, sizes, bits, min_face_needed, min_surf_needed


def horizontal_slivers(labels,ids,side=4,*,whole_surface=False,top_only=False):
    """World coordinates neutral to a diagnostic score, using only the supplied reference view."""
    if side not in SIDES:
        raise ValueError('Choose a tested native square side')
    _,_,_,bits,face_side,surface_side=block_footprints(labels,ids)
    candidates=bits==1 if top_only else (bits==1)|(bits==2)
    optional=candidates & ((surface_side if whole_surface else face_side)<side)
    return {tuple(b['xyz_world']) for b in labels['blocks'] if optional[b['id']]}


def wall_coverage(parts,camera,size,world_axis):
    """Union of native samples positively assigned a vertical orientation by at least one mask."""
    p=np.asarray(camera['P'],float)
    p=p/np.linalg.norm(p[2,:3])
    focal=np.linalg.norm(p[0,:3]-.5*size[0]*p[2,:3])
    rotation=np.linalg.solve(intrinsic(size,focal),p[:,:3])
    wall=np.zeros(size[::-1],bool)
    for part in parts:
        normal=part.get('normal_camera')
        if normal is None or world_axis[np.argmax(abs(rotation.T@normal))]==1:
            continue
        patch=_native_alpha(part['support'],size)
        if patch is not None:
            mask,(l,t,r,b)=patch
            wall[t:b,l:r]|=np.asarray(mask,bool)
    return wall


def score(job):
    run,split,frame,inputs,row,out = job
    image,parts,counts,camera,pose,labels,ids,truth,placed = view(run,split,frame,inputs,row)
    blocks=labels['blocks']
    n=len(blocks)+1
    owners,axes,sizes,bits,min_face_needed,min_surf_needed=block_footprints(labels,ids)
    group_masks={'top_only':bits==1,'underside_only':bits==2,'any_wall':(bits&4)>0}
    assert sum(int(v.sum()) for v in group_masks.values())==len(blocks)
    real=np.arange(n)>0
    hit=np.array([False]+[tuple(b['xyz_world']) in placed for b in blocks])
    extras=placed-truth
    assert int(hit.sum())==row['relative_geometry']['correct'] and len(extras)==row['relative_geometry']['extras']
    owner,_,world_axis,world=projection(camera,pose,sorted(map(tuple,read(run/split/frame['id']/'cells.json')['local'])),
                                          row['translation'],ids.shape)
    assert set(map(tuple,world))==placed
    extra_indices=[i for i,p in enumerate(world) if tuple(p) in extras]
    pixels=owners[ids]
    stage=stage_evidence(parts,counts,image.size,camera,world_axis,axes[ids])
    nstage=tally(pixels.ravel(),(stage>=1).ravel(),n)
    naxis=tally(pixels.ravel(),(stage>=2).ravel(),n)
    ncovered=tally(pixels.ravel(),(owner>=0).ravel(),n)
    cohort=(bits>0)&(bits<4)&~hit&(ncovered*2<sizes)&(nstage*2>=sizes)&(naxis*2<sizes)
    positive_wall=wall_coverage(parts,camera,image.size,world_axis)
    nwall=tally(pixels.ravel(),positive_wall.ravel(),n)
    exclusive=tally(pixels.ravel(),(positive_wall & (stage<2)).ravel(),n)
    total_by_owner=np.bincount(owner[owner>=0],minlength=len(world))
    extra_rays=np.isin(owner,extra_indices)
    extra_owner=owner[extra_rays]
    extra_truth=pixels[extra_rays]
    def pack(mask):
        # Extras stay errors in every score. Separately attribute projected extras
        # only when a strict majority of their own visible pixels land in this region.
        on=np.bincount(extra_owner[mask[extra_truth]],minlength=len(world))
        return [int(np.count_nonzero(mask&hit)),int(np.count_nonzero(mask&~hit)),
                int(np.count_nonzero(mask&cohort)),
                sum(int(2*on[i]>total_by_owner[i]) for i in extra_indices)]
    stats={'strict':pack(real&False),'facade_only':pack((bits==1)|(bits==2)),
           'any_top_only':pack(bits==1),'tiny_area_under_16':pack(real&(sizes<16))}
    for side in SIDES:
        for label,measure in [('surface',min_surf_needed),('face',min_face_needed)]:
            thin=real&(measure<side)
            stats[f'{label}_{side}_all']=pack(thin)
            stats[f'{label}_{side}_horizontal']=pack(thin&((bits==1)|(bits==2)))
            for name,gmask in group_masks.items():
                stats[f'{label}_{side}_{name}']=pack(thin&gmask)
    # Count distinct frontmost faces on each missed and correctly placed cube.
    nf=np.bincount(owners[1:],minlength=n)
    for j in (1,2,3):
        stats[f'{j}_visible_faces']=pack(nf==j)
    if 'colonnade-01' in frame['id'] and split!='extra':
        figure(out/'sliver-colonnade.jpg',image,ids,owners,hit,group_masks['top_only'],
               group_masks['top_only']&(min_surf_needed<8),group_masks['top_only']&(min_face_needed<4))
    wall_masks={}
    for description,selection in [('all',cohort),('local4_required',cohort&(min_face_needed>=4))]:
        wall_masks[description]=dict(total=int(selection.sum()),
            wall_at_least_half=int(np.count_nonzero(selection & (nwall*2>=sizes))),
            wall_no_horizontal_at_least_half=int(np.count_nonzero(selection & (exclusive*2>=sizes))),
            no_vertical_mask_pixels=int(np.count_nonzero(selection & (nwall==0))))
    return dict(id=frame['id'],split=split,size=image.size,pack_sha256=row['pack_sha256'],
                rgb_sha256=row['rgb_sha256'],correct=int(hit.sum()),missed=len(truth-placed),extras=len(extras),
                horizontal_mask_cohort=int(cohort.sum()),cohort_wall_masks=wall_masks,stats=stats)


def aggregate(rows):
    counts=Counter()
    for row in rows:
        counts.update({k:row[k] for k in ('correct','missed','extras','horizontal_mask_cohort')})
    output={}
    for k in rows[0]['stats']:
        matched,missed,cohort,extra=(sum(col) for col in zip(*(r['stats'][k] for r in rows)))
        output[k]=dict(matched=matched,missed=missed,cohort=cohort,projected_extras=extra,
            optional_f1=rates(counts['correct']-matched,counts['missed']-missed,counts['extras'])['f1'])
    walls={group:{key:sum(r['cohort_wall_masks'][group][key] for r in rows) for key in fields}
           for group,fields in rows[0]['cohort_wall_masks'].items()}
    return dict(counts,views=len(rows),cohort_wall_masks=walls,rules=output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('../visible-error-budget/runs/primary-production'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/strip-orientation-01'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=read(args.run/'summary.json')
    manifest=checked(args.run/'cases.json',meta['cases_sha256'])
    rows={(r['split'],r['id']):r for r in read(manifest)}
    jobs=[(args.run,s,f,h,rows[s,f['id']],args.out) for s,f,h in frames.cases([*frames.EVALUATION,'extra'])
          if s!='extra' or 'colonnade-01' in f['id']]
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(score,jobs))
    total=aggregate([r for r in results if r['split']!='extra'])
    assert total['views']==104
    assert all(total[k]==meta['groups']['fixed'][k] for k in ('correct','missed','extras'))
    families={r['id']:aggregate([r]) for r in results if 'colonnade-01' in r['id']}
    save(args.out/'sliver-counts.json',dict(production_cases_sha256=digest(manifest),fixed=total,
         colonnade=families,columns=['matched','missed','cohort','projected_extras'],frames=results))


if __name__=='__main__':
    main()
