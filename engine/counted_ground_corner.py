"""Ground two fully visible, counted perpendicular wall faces at their common corner."""
from itertools import combinations, product

import numpy as np

from engine.pillar_seams import orientation
from engine.visible_raster import SAMPLE_STRIDE, rasterize


def agreement(projection, mask, axis, sign, level, corner, across, height):
    free = 2 - axis
    quad = np.zeros((4, 3))
    quad[:, axis] = level
    quad[:, free] = [corner, corner + across, corner + across, corner]
    quad[:, 1] = [0, 0, height, height]
    camera = quad @ projection[:, :3].T + projection[:, 3]
    if camera[:, 2].min() <= 1.:
        return 0.
    camera[:, :2] /= SAMPLE_STRIDE
    samples = rasterize(camera, mask.shape[1], mask.shape[0])
    if samples is None:
        return 0.
    box, pixels, *_ = samples
    common = np.count_nonzero(pixels & mask[box])
    return common / (int(mask.sum()) + int(pixels.sum()) - common)


def grounded_pairs(image, charts, parts, masks, P):
    candidates = []
    for i, (chart, part) in enumerate(zip(charts, parts, strict=True)):
        counts = chart['pitch']['counts']
        if not counts or chart['quad'] is None or part.get('normal_camera') is None:
            continue
        size = np.rint(counts).astype(int)
        if min(size) < 2 or max(abs(size - counts)) > .2:
            continue
        candidates.append((i, size, np.asarray(chart['quad'], float)))
    if len(candidates) < 2:
        return {}
    p, rot = orientation(dict(P=P), image.size)
    inv = np.linalg.inv(p[:, :3])
    eye = -inv @ p[:, 3]
    result = {}
    for (i, dims, quad), (j, other, second) in combinations(candidates, 2):
        if dims[1] != other[1]:
            continue
        normals = [rot.T @ parts[k]['normal_camera'] for k in (i, j)]
        axes = [int(np.argmax(abs(n))) for n in normals]
        if set(axes) != {0, 2}:
            continue
        signs = [int(np.sign(n[a])) for a,n in zip(axes,normals)]
        # Both top and bottom endpoints must meet on the same projected edge.
        edges = [(a,b) for a,b in product((0,1), repeat=2)
                 if max(np.linalg.norm(quad[a]-second[b]),np.linalg.norm(quad[3-a]-second[3-b])) <= 6.]
        if len(edges) != 1:
            continue
        a,b = edges[0]
        bottom = .5*(quad[3-a]+second[3-b])
        ray = inv @ [*bottom,1.]
        if ray[1] >= -1.e-7:
            continue
        point = eye - ray*(eye[1]/ray[1])
        levels = np.rint(point[axes]).astype(int)
        fits = []
        # Rectified chart endpoints may be a few native pixels beyond the mask.
        # Test neighboring integer intersections against both entire measured masks.
        for da, db in product((-1,0,1), repeat=2):
            first, second_level = levels + [da,db]
            left = agreement(p,masks[i],axes[0],signs[0],first,second_level,-signs[1]*dims[0],dims[1])
            right = agreement(p,masks[j],axes[1],signs[1],second_level,first,-signs[0]*other[0],other[1])
            if min(left, right) >= .90:
                fits.append((int(first),int(second_level),left,right))
        if len(fits) == 1:
            for k,axis,level,sign in zip((i,j),axes,fits[0][:2],signs):
                result[k]=(axis,level,sign)
    return result

def edits(image, parts, counts, status, placed):
    """Confine changes to incompatible strict chart seeds and the reprojected wall faces."""
    from engine.visible_grid import _blocks, _carriers, _cell_map
    from engine.visible_placement import _charts, _frontmost_cells, _sample_face
    from engine.visible_raster import observed_support

    if not placed or status.get('P') is None:
        return set(), set(), {}
    charts = _charts(image, '26.3', parts, counts)
    shape = tuple((n + (SAMPLE_STRIDE - 1)//2)//SAMPLE_STRIDE for n in image.size[::-1])
    masks = [observed_support(p['support'], shape,image.size) if p.get('support')
             else np.zeros(shape, bool) for p in parts]
    projection=np.asarray(status['P'],float)
    anchors=grounded_pairs(image,charts,parts,masks,projection)
    if not anchors:
        return set(), set(), {}
    chosen={r['id']:(r['axis'],r['level']) for r in status.get('plane_offsets',[])}
    if any(chosen.get(parts[i]['id']) != plane[:2] for i,plane in anchors.items()):
        return set(),set(),{}
    carriers = _carriers(image,charts,masks,projection)
    by_mask={id(m): plane for m, plane, _ in carriers}
    incompatible = {i for i,plane in anchors.items() if by_mask.get(id(masks[i])) not in (None,plane)}
    if not incompatible:
        return set(),set(),{}
    cache={}
    bad,_ ,_ = _blocks([masks[i] for i in incompatible],carriers,projection,cache)
    keep,_,_ = _blocks([m for i,m in enumerate(masks) if i not in incompatible],carriers,projection,cache)
    candidates={}
    for i,plane in anchors.items():
        axis,level,sign=plane
        grid=_cell_map(projection,shape,plane)
        mask=masks[i]
        covered=np.bincount(grid.inverse,weights=mask.ravel()[grid.valid_pixels],minlength=len(grid.cells))
        for coord in grid.cells[(covered >= 4)&(covered >= .75*grid.counts)]:
            cell=np.zeros(3,int)
            cell[axis]=level-int(sign>0)
            cell[[k for k in range(3) if k!=axis]]=coord
            if cell[1]>=0:
                candidates.setdefault(tuple(cell.tolist()),[]).append(i)
    remove = (placed & bad) - keep - candidates.keys()
    trial=sorted((placed - remove) | candidates.keys())
    eye=-np.linalg.solve(projection[:,:3],projection[:,3])
    owner,axes=_frontmost_cells(trial,projection,eye,shape,face_axes=True)
    index={cell:k for k,cell in enumerate(trial)}
    added=set()
    for cell in candidates.keys()-placed:
        for i in candidates[cell]:
            axis,_,sign=anchors[i]
            hit=_sample_face(projection,cell,axis,sign,shape)
            if hit is None: continue
            box,pixels,*_=hit
            votes=(owner[box]==index[cell]) & (axes[box]==axis) & masks[i][box] & pixels
            if votes.sum() >= max(4,.75*pixels.sum()):
                added.add(cell)
                break
    return added,remove,{parts[i]['id']:plane for i,plane in anchors.items()}

