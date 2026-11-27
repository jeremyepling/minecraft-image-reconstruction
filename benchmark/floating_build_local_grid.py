"""A build-owned lattice from native sprite witnesses, independent of ground and horizon."""
from collections import deque
import os
os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")

import cv2
import numpy as np
from engine.atlas_count import _shortlist
from engine.geometry import project, transform, UNIT
from engine.surface_model import intrinsic
from engine.visible_raster import observed_support
from benchmark.full_block_lattice import atlas, consensus, line_camera, mask_parts, sprite_candidates, verify_sprite
from benchmark.camera_structure_scaffold import outline_lines


def count_options(row):
    counts = row.get("counts", [None, None])
    if len(counts) != 2 or any(v is None or not np.isfinite(v) or v < .95 for v in counts):
        return []
    # A texture may repeat inside the full cube. Test octave alternatives against the vanilla atlas.
    return [np.asarray(counts) * ratio for ratio in (1., .5, 2.) if max(counts)*ratio <= 60]


def chart_projection(chart, size):
    """Lift the photographed wall's x/right and y/up tangent directions for evaluator compass registration."""
    k = intrinsic(size, size[1])
    u, v = np.linalg.solve(k, chart[:, :2]).T
    z = np.cross(u, -v)
    if np.linalg.norm(z) < 1e-12:
        return None
    z *= np.sqrt(np.linalg.norm(u)*np.linalg.norm(v)) / np.linalg.norm(z)
    return np.column_stack((chart[:, 0], -chart[:, 1], k@z, chart[:, 2]))


def planar_charts(rgb, part, row, size):
    normal = part.get("normal_camera")
    quad = row.get("quad")
    if not quad or normal is None or abs(normal[1]) > .63:
        return
    quad = np.asarray(quad)
    across = (quad[1]-quad[0]+quad[2]-quad[3])/2
    down = (quad[3]-quad[0]+quad[2]-quad[1])/2
    if across[0] <= abs(across[1]) or down[1] <= abs(down[0]):
        return
    support = observed_support(part["support"], (size[1]//4, size[0]//4), size)
    lib = atlas()
    for counts in count_options(row):
        chart = transform(quad)@np.diag([1/counts[0], 1/counts[1], 1])
        # Rectify at eight samples per block, search all possible sprite phases.
        w,h = np.ceil(counts*8).astype(int)
        if min(w,h) < 8:
            continue
        x,y = np.meshgrid(np.arange(w),np.arange(h))
        uv = np.stack((x+.5,y+.5),axis=-1)/8
        xy = project(chart, uv).astype('f')
        patch = cv2.remap(rgb,xy[:,:,0]-.5,xy[:,:,1]-.5,cv2.INTER_LINEAR)
        mask = cv2.remap(support.astype('B'),xy[:,:,0]/4-.5,xy[:,:,1]/4-.5,cv2.INTER_NEAREST)
        sums = cv2.integral(mask)
        inside = (sums[8:,8:]-sums[8:,:-8]-sums[:-8,8:]+sums[:-8,:-8]) == 64
        if not inside.any():
            continue
        keys,loss = _shortlist(patch,mask.astype(bool),lib)
        gray = patch.mean(2).astype('f')
        seeds = []
        for key in keys[:8]:
            if loss[key]>20 or lib.spreads[key].mean()<2.5:
                continue
            for tile in lib.patterns[key]:
                compact = cv2.resize(tile,(8,8),interpolation=cv2.INTER_AREA)
                corr = cv2.matchTemplate(gray,compact,cv2.TM_CCOEFF_NORMED)
                corr[~inside]=-1
                _,peak,_,(i,j) = cv2.minMaxLoc(corr)
                if peak >= .69:
                    seeds.append((peak,(i,j)))
        seen_phases = set()
        for peak,(i,j) in sorted(seeds,reverse=True)[:14]:
            phase = (i%8,j%8)
            if phase in seen_phases:
                continue
            seen_phases.add(phase)
            todo = deque([(i,j)])
            tried = set()
            accepted = []
            while todo:
                x,y = todo.popleft()
                if (x,y) in tried or not (0 <= y < inside.shape[0] and 0 <= x < inside.shape[1]):
                    continue
                tried.add((x,y))
                if not inside[y,x]:
                    continue
                corners = project(chart, UNIT + np.array([x,y])/8)
                if min(np.linalg.norm(corners-np.roll(corners,1,axis=0),axis=1)) < 7:
                    continue
                hit = verify_sprite(rgb,corners)
                if not hit or hit[2]<.68:
                    continue
                accepted.append(dict(cell=[(x-i)//8,-(y-j)//8-1,0],quad=corners.tolist(),
                                     block=hit[0],texture=hit[1],corr=hit[2],surface=part["id"]))
                todo.extend(((x-8,y),(x+8,y),(x,y-8),(x,y+8)))
            if len(accepted)>=2 or (accepted and accepted[0]['corr']>=.83):
                # This is an observed sheet; its two tangents alone do not register other sheets.
                translation = np.array([[1, 0, i/8], [0, -1, j/8], [0, 0, 1]])
                yield accepted, chart @ translation


def planar_witnesses(rgb, part, row, size):
    for accepted, chart in planar_charts(rgb, part, row, size):
        # Preserve the historical up-oriented evaluator transform.
        projection = chart_projection(chart @ np.diag([1, -1, 1]), size)
        if projection is not None:
            yield accepted, projection


def native_sheet(rgb, part, anchors, linear):
    """Keep the strongest connected, photographed wall when global multi-axis agreement fails."""
    best = []
    best_p = None
    axis = part['axis']
    free = [i for i in range(3) if i != axis]
    for seed in sorted(anchors, key=lambda a:-a['corr'])[:3]:
        corners = np.asarray(seed['xyz'])
        origin = corners[0]
        eye = -origin
        projection = np.column_stack((linear, linear@origin))
        todo = deque([(0,0)])
        tried = set()
        accepted = []
        while todo:
            i,j = todo.popleft()
            if (i,j) in tried:
                continue
            tried.add((i,j))
            xyz = np.zeros((4,3))
            xyz[:,free] = np.array([i,j])+UNIT
            quad = project(projection,xyz)
            if not np.isfinite(quad).all() or min(np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1))<7:
                continue
            # Exact mask support at its source resolution for the entire visible footprint.
            grid = quad/4-.5
            if np.min(grid)<0 or np.any(grid.max(0)>=part['mask'].shape[::-1]):
                continue
            lo,hi = np.floor(grid.min(0)).astype(int),np.ceil(grid.max(0)).astype(int)+1
            cover=np.zeros((hi[1]-lo[1],hi[0]-lo[0]),'B')
            cv2.fillConvexPoly(cover,np.rint(grid-lo).astype('i'),1)
            support=part['mask'][lo[1]:hi[1],lo[0]:hi[0]]
            if support.shape!=cover.shape or not support[cover.astype(bool)].all():
                continue
            hit=verify_sprite(rgb,quad)
            if not hit or hit[2]<.68:
                continue
            cell=np.zeros(3,dtype=int)
            cell[free]=[i,j]
            cell[axis]=-int(eye[axis]>0)
            accepted.append(dict(cell=cell.tolist(),quad=quad.tolist(),block=hit[0],texture=hit[1],
                                 corr=hit[2],surface=part['id']))
            todo.extend(((i-1,j),(i+1,j),(i,j-1),(i,j+1)))
        if (len(accepted)>=2 or (accepted and accepted[0]['corr']>=.83)) and len(accepted)>len(best):
            best,best_p=accepted,projection
    return best,best_p


def predict(image, parts, counts, *, native_fallback=False):
    rgb = np.asarray(image.convert("RGB"))
    rows = {r["id"]:r for r in counts['surfaces']}
    walls = [p for p in parts if p.get("support") and p.get("normal_camera") is not None
             and abs(p["normal_camera"][1]) <= .63]
    lines,_ = outline_lines(image,walls,[])
    linear = line_camera(lines,image.size)
    native = []
    consensus_rows = []
    fitted_parts = []
    if linear is not None:
        fitted_parts = mask_parts(walls,image.size,linear)[:8]
        for part in fitted_parts:
            if part['axis'] != 1:
                native.extend(sprite_candidates(part,rgb,linear))
        consensus_rows, eye, _ = consensus(native)
        strong_axes = {r['axis'] for r in consensus_rows if sum(other['surface']==r['surface']
                       for other in consensus_rows)>=2}
        if len(strong_axes)>=2:
            P = np.column_stack((linear,-linear@eye))
            for r in consensus_rows:
                c = np.array(r['xyz']).mean(0)+eye
                axis = r['axis']
                c[axis]=round(c[axis])-.02*np.sign(eye[axis]-c[axis])
                r['cell']=np.floor(c).astype(int).tolist()
            return consensus_rows,P,dict(status='supported_nonempty',local_grid=True,global_projection=True,
                                          native=len(native),reason='two perpendicular repeated native sprite sheets')
    best = []
    projection = None
    if native_fallback:
        for part in fitted_parts:
            local = [a for a in native if a['surface']==part['id']]
            found,p = native_sheet(rgb,part,local,linear)
            if len(found)>len(best):
                best,projection=found,p
    for part in sorted(walls,key=lambda p:-p['support_area'])[:16]:
        if part['id'] not in rows:
            continue
        for verified,p in planar_witnesses(rgb,part,rows[part['id']],image.size):
            if (len(verified), np.mean([a['corr'] for a in verified])) > (
                    len(best), np.mean([a['corr'] for a in best]) if best else 0):
                best,projection = verified,p
    return best,projection,dict(status='local_only_nonempty' if best else 'no_local_grid',
                                global_projection=False,local_grid=bool(best),native=len(native),
                                axis_lines=len(lines),three_axes=linear is not None,
                                reason='verified native wall sprite chart' if best else 'no certified native wall chart')
