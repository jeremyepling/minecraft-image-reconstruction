"""Measure photographed return edges on a visible wall mask before atlas or grid fitting."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_depth_before_atlas as pink
from benchmark import floating_package_rechecks as package

REVIEW = pink.ROOT / 'benchmark/reviews/floating-pregrid-visible-returns-01'
OUT = pink.ROOT / 'runs/pregrid-visible-returns'
# Previously reviewed Stable Yard is development. Other families never set thresholds.
TARGETS = {
    'train-stable-yard-1-warm-high-right': 'mask-0',
    'train-stable-yard-1-warm-low-left': 'mask-0',
    'train-pillars-01-marble-low-left': 'mask-1',
    'train-viaduct-02-riverstone-high-right': 'mask-0',
    'test-castle-01-ashlar-high-right': 'mask-0',
}


def photographed_cuts(rgb, mask, normal, depth, base):
    """Split only where the finite observed line disconnects the original piece."""
    ids = base.copy()
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contour = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), .9), 30, 85)
    contour[~mask] = 0
    lines = cv2.HoughLinesP(contour, 1, np.pi/360, 24, minLineLength=26, maxLineGap=6)
    proposals = []
    for x, y, u, v in (() if lines is None else lines[:, 0]):
        start = np.array([x, y])
        delta = np.array([u-x, v-y])
        length = np.linalg.norm(delta)
        tangent = delta / length
        perpendicular = np.array([-tangent[1], tangent[0]])
        xy = start + np.arange(2, length-1, 2)[:, None]*tangent
        probes = np.rint(xy[:, None] + np.array([-12,-8,-4,4,8,12])[None,:,None]*perpendicular).astype(int)
        valid = ((probes >= 0) & (probes < mask.shape[::-1])).all(axis=(1,2))
        xx, yy = probes[valid].transpose(2,0,1)
        if len(xx) < 12:
            continue
        labels = base[yy, xx]
        for parent in np.unique(labels[:,2]):
            if not parent:
                continue
            owned = (labels == parent).all(axis=1)
            if owned.sum() < max(12, .55*len(xy)):
                continue
            p = depth[yy[owned], xx[owned]]
            # At each crossing, extrapolate the two side-specific slopes to the seam.
            left = p[:,2] + .5*(p[:,2]-p[:,0])
            right = p[:,3] + .5*(p[:,3]-p[:,5])
            jump = right-left
            med = np.median(jump)
            noise = max(1.e-6, .004*np.median(np.abs(p)),
                        np.median(abs(p[:,0]-2*p[:,1]+p[:,2]) + abs(p[:,3]-2*p[:,4]+p[:,5])))
            consistent = np.mean(np.sign(jump) == np.sign(med))
            if abs(med) < noise or consistent < .80:
                continue
            probes_normal = normal[yy[owned], xx[owned]]
            agreement = np.median(np.sum(probes_normal[:,1] * probes_normal[:,4], axis=1))
            if agreement < .93:
                continue
            proposals.append(dict(parent=int(parent), start=start.tolist(), end=[int(u),int(v)],
                                  strength=float(abs(med)/noise), sign_agreement=float(consistent),
                                  probes=int(owned.sum()), jump=float(med)))
    accepted = []
    for proposal in sorted(proposals, key=lambda p:-p['probes']):
        a,b = np.array(proposal['start']), np.array(proposal['end'])
        t = (b-a)/np.linalg.norm(b-a)
        region_ids = ids[tuple(np.rint((a+b)/2).astype(int)[::-1])]
        if not region_ids:
            continue
        region = ids == region_ids
        if not np.all(base[region] == proposal['parent']):
            continue
        yy, xx = np.nonzero(region)
        origin = np.array([xx.min(), yy.min()])
        box = np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
        remaining = region[box].astype('uint8')
        cv2.line(remaining, tuple(np.rint(a-origin-3*t).astype(int)),
                 tuple(np.rint(b-origin+3*t).astype(int)), 0, 3)
        count, islands, stats, _ = cv2.connectedComponentsWithStats(remaining, connectivity=4)
        if count != 3 or min(stats[1:,cv2.CC_STAT_AREA]) < max(80, .025*len(xx)):
            continue
        # Assign removed line pixels to one of its two observed neighbors.
        dist1 = cv2.distanceTransform((islands!=1).astype('uint8'),cv2.DIST_L2,5)
        dist2 = cv2.distanceTransform((islands!=2).astype('uint8'),cv2.DIST_L2,5)
        ids[box][region[box] & (dist2 < dist1)] = int(ids.max())+1
        accepted.append(proposal)
    assert np.array_equal(ids>0, mask)
    return ids, accepted, len(proposals)


def native(case, ids, *, reference_control=False):
    """Score internal same-direction depth and material transitions on native owners."""
    frame = Path(case['frame'])
    raw = pink.read(frame/'labels.json')
    with np.load(frame/'labels.npz') as z:
        front = z['face_id']
    blocks = {b['id']:b['block_id'] for b in raw['blocks']}
    materials, directions, planes = {}, {}, {}
    table = np.zeros((int(front.max())+1,3),np.int32)
    for f in raw['faces']:
        offset = round(float(np.median(np.array(f['corners_world']) @ f['normal_world'])), 3)
        for k, (mapping, key) in enumerate(((materials,blocks[f['block']]),(directions,f['direction']),(planes,(f['direction'],offset)))):
            table[f['id'],k] = mapping.setdefault(key,len(mapping)+1)
    material, direction, plane = table[front].transpose(2,0,1)
    if reference_control:
        ids = np.where(ids>0, plane+1, 0)
    inside = (ids>0) & (front>0)
    edge = pink.mask_boundary(ids, inside)
    near = cv2.distanceTransform((~edge).astype('uint8'), cv2.DIST_L2,5) <= 3
    counts = Counter()
    for a,b in [(np.s_[1:], np.s_[:-1]),(np.s_[:,1:], np.s_[:,:-1])]:
        relief = inside[a]&inside[b]&(plane[a]!=plane[b])&(direction[a]==direction[b])
        for name, selected in [('parallel',relief),('same_material_parallel',relief&(material[a]==material[b]))]:
            counts[name] += int(selected.sum())
            counts[name+'_found'] += int((selected&(near[a]|near[b])).sum())
    stats = pink.ownership(case, ids)
    counts.update(stats['edges'])
    counts.update(dominant=stats['dominant'],pixels=stats['pixels'],parts=len(stats['parts']))
    return dict(counts)


def run(job):
    panel, case = job
    cv2.setNumThreads(1)
    name = case['id']
    plane_id = TARGETS[name]
    src = pink.data(panel)[0]
    source = Path(case['variants']['sky']['image'])
    assert pink.digest(source) == case['variants']['sky']['source_sha256']
    rgb = np.array(Image.open(source).convert('RGB'))
    normal, depth, hashes = pink.source_fields(panel, case, src, rgb.shape[1::-1])
    path = src/'sky'/name/'source-handoff/surfaces.json'
    plane = next(p for p in pink.read(path)['planes'] if p['id']==plane_id)
    mask = package.native_mask(plane, rgb.shape[1::-1])
    coarse, geometry = pink.partition(rgb,mask,normal,depth)
    base, _, _ = pink.refine_adjacent(rgb,mask,normal,depth,coarse,geometry)
    ids, accepted, proposed = photographed_cuts(rgb,mask,normal,depth,base)
    from benchmark.floating_pregrid_dense_contacts import dense_cuts
    dense, dense_edges = dense_cuts(base,pink.prior.partition(rgb,mask,depth),depth,normal)
    assert np.array_equal(dense>0,mask)
    folder = OUT/panel/'source'/name
    folder.mkdir(parents=True,exist_ok=True)
    output = folder/(plane_id+'-ids.png')
    Image.fromarray(ids).save(output)
    Image.fromarray(base).save(folder/(plane_id+'-baseline.png'))
    Image.fromarray(ids).save(folder/(plane_id+'-returns.png'))
    Image.fromarray(dense).save(folder/(plane_id+'-dense.png'))
    row = dict(name=name,panel=panel,plane=plane_id,source=pink.digest(source),support=pink.digest(path),fields=hashes,
               ids=pink.digest(output),baseline_ids_equal=bool(np.array_equal(base,ids)),lines=proposed,accepted=accepted,
               dense_edges=dense_edges,dense_ids_equal=bool(np.array_equal(base,dense)),
               oracle=native(case, base, reference_control=True),
               scores={key:native(case,field) for key,field in [('coarse',coarse),('baseline',base),('returns',ids),('dense',dense)]})
    if accepted:
        from PIL import ImageDraw
        # Draw only after source decisions and native scoring are complete.
        aa,bb=np.asarray(accepted[0]['start']),np.asarray(accepted[0]['end'])
        mid=(aa+bb)//2
        left,top=mid-[95,65]
        box=np.s_[top:top+130,left:left+190]
        before=pink.mask_boundary(base,mask)
        after=pink.mask_boundary(ids,mask)
        frame=Path(case['frame'])
        with np.load(frame/'labels.npz') as labels:
            face=labels['face_id']
        planes=np.zeros(int(face.max())+1,'int32')
        keys={}
        for f in pink.read(frame/'labels.json')['faces']:
            z=round(float(np.median(np.array(f['corners_world'])@f['normal_world'])),3)
            planes[f['id']]=keys.setdefault((f['direction'],z),len(keys)+1)
        true=pink.mask_boundary(planes[face],mask)
        sheet=Image.new('RGB',(760,292),'white')
        draw=ImageDraw.Draw(sheet)
        for i,(title,edges,color) in enumerate([('Added source cut',after & ~before,(40,250,220)),
                                                ('Native visible-plane boundaries',true,(255,90,155))]):
            crop=rgb[box].copy()
            crop[edges[box]]=color
            sheet.paste(Image.fromarray(crop).resize((380,260),Image.Resampling.NEAREST),(380*i,32))
            draw.text((380*i+9,9),title,fill='black')
        sheet.save(REVIEW/(name+'-cut.png'))
    pink.save(REVIEW/(name+'.json'),row)
    print(name, len(accepted), proposed, row['scores']['returns'],flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--build',action='store_true')
    args=cli.parse_args()
    work=[(g,c) for g,c in pink.jobs() if c['id'] in TARGETS and g in ('sky','holdout3')]
    assert len(work)==len(TARGETS)
    if args.build:
        from benchmark.floating_pregrid_returns_replay import replay
        for g,c in work:
            if not pink.read(REVIEW/(c['id']+'.json'))['baseline_ids_equal']:
                replay(g,c)
    else:
        with ProcessPoolExecutor(3) as pool:
            list(pool.map(run,work))
