
"""Assemble independently measured visible face grids at shared image edges.

Inputs are per-plane image homographies and integer face cells. Neither fitting
nor export uses a full camera, scene distance, world origin, or ground plane.
"""
from collections import defaultdict, deque
from itertools import combinations, product
from math import floor

import numpy as np


AXES = np.eye(3, dtype=int)


def axis_vector(code):
    if not isinstance(code, int) or not 1 <= abs(code) <= 3:
        raise ValueError(f'Invalid signed axis: {code}')
    return AXES[abs(code)-1] * (1 if code > 0 else -1)


def project(h, uv):
    q = h @ np.array((*uv, 1.))
    if not np.isfinite(q).all() or abs(q[2]) < 1e-9:
        return None
    return q[:2] / q[2]


def _walls(charts):
    """Choose a shared compass from a wall normal and vertical image lines."""
    walls = [c for c in charts if c.get('cells') and c.get('normal_camera') is not None
             and abs(c['normal_camera'][1]) < .67]
    if not walls:
        return [], None
    anchor = max(walls, key=lambda c: (len(c['cells']), c.get('confidence', 0), c['id']))
    norm = np.asarray(anchor['normal_camera'])
    forward = norm.copy()
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    up = np.array([0., -1., 0.])
    basis = np.array([np.cross(up, forward), up, forward])
    result = []
    for c in walls:
        n = np.asarray(c['normal_camera'])
        dots = basis @ n
        normal = int(np.argmax(np.abs(dots)))
        if normal == 1 or abs(dots[normal]) < .65:
            continue
        h = np.asarray(c['H'], float)
        center = np.median(np.array([(p['u'], p['v']) for p in c['cells']]), axis=0) + .5
        at = project(h, center)
        around = [project(h, center+np.eye(2)[axis]) for axis in range(2)]
        if at is None or any(end is None for end in around):
            continue
        tangents = np.array([end-at for end in around])
        lengths = np.linalg.norm(tangents, axis=1)
        if min(lengths) < 1e-6:
            continue
        verticality = abs(tangents[:,1])/lengths
        which_up = int(np.argmax(verticality))
        which_side = 1-which_up
        horizontal = 2-normal
        if (verticality[which_up] < .75 or verticality[which_up]-verticality[which_side] < .2
                or abs(basis[horizontal,0]) < .12):
            continue
        codes = [0, 0]
        codes[which_up] = 2 if tangents[which_up,1] < 0 else -2
        codes[which_side] = ((horizontal+1) if tangents[which_side,0]*basis[horizontal,0] > 0
                             else -(horizontal+1))
        signed_normal = (normal+1) if dots[normal] > 0 else -(normal+1)
        result.append(dict(c, axes=codes, normal_axis=signed_normal))
    reference = next((c for c in result if c['id']==anchor['id']), None)
    hint = None if reference is None else dict(normal_camera=anchor['normal_camera'],
                                                local_normal_axis=reference['normal_axis'])
    return result, hint


def oriented(charts):
    known = [c for c in charts if c.get('axes') and c.get('normal_axis')]
    if len(known)==len(charts):
        return known, None
    if known:
        raise ValueError('Use a shared orientation contract across the input charts')
    return _walls(charts)


def _edge_depth(h, uv_midpoint, uv_center, field):
    """Extrapolate an ordinal image-depth estimate to a face's boundary."""
    if field is None:
        return None
    d = np.array(uv_center)-uv_midpoint
    positions = [project(h, uv_midpoint+fraction*d) for fraction in (.36,.84)]
    readings=[]
    for position in positions:
        if position is None:
            return None
        x,y = np.rint(position-.5).astype(int)
        if not (0<=y<field.shape[0] and 0<=x<field.shape[1]):
            return None
        value=float(field[y,x])
        if not np.isfinite(value) or value<=0:
            return None
        readings.append(value)
    near,deep=readings
    gradient=deep-near
    return near-(.36/.48)*gradient, abs(gradient)


def _distant_depth_edges(a, b):
    if a is None or b is None:
        return False
    discontinuity=abs(a[0]-b[0])
    local_noise=max(.045*(abs(a[0])+abs(b[0]))*.5, 2.0*(a[1]+b[1]))
    return discontinuity>local_noise


def face_boundary(chart, relative_depth=None):
    """Return 2D image edge samples paired with unshifted 3D integer corners."""
    h = np.asarray(chart['H'], float)
    vectors = np.array([axis_vector(a) for a in chart['axes']])
    if len({abs(a) for a in (*chart['axes'], chart['normal_axis'])})!=3:
        raise ValueError(f'Chart axes are not perpendicular: {chart["id"]}')
    cells = {(row['u'],row['v']):row for row in chart['cells']}
    result = []
    for u,v in sorted(cells):
        for index, neighbor, left, right in (
            (3,(u-1,v),(u,v),(u,v+1)), (1,(u+1,v),(u+1,v),(u+1,v+1)),
            (0,(u,v-1),(u,v),(u+1,v)), (2,(u,v+1),(u,v+1),(u+1,v+1)),
        ):
            visible = cells[u,v].get('visible_edges')
            if neighbor in cells or (visible is not None and index not in visible):
                continue
            a,b = np.array([left,right]) @ vectors
            direction = int(np.argmax(np.abs(b-a)))
            flags = cells[u,v].get('edge_samples')
            bins = None if flags is None else flags[index]
            if a[direction]>b[direction]:
                if bins is not None:
                    bins = sum(((bins >> k) & 1) << (15-k) for k in range(16))
                a,b,left,right=b,a,right,left
            start,end = project(h,left),project(h,right)
            if start is None or end is None:
                continue
            length = float(np.linalg.norm(end-start))
            if length < 4:
                continue
            mid = (np.asarray(left)+right)*.5
            sample = _edge_depth(h,mid,(u+.5,v+.5),relative_depth)
            result.append((start,end,a,b,direction,length,bins,sample))
    return result


def propose_joins(charts, min_edges=2, relative_depth=None, edge_pixels=3.5):
    """Group agreeing shared edges into integer translation hypotheses."""
    by_edge = defaultdict(list)
    edges = [face_boundary(c,relative_depth) for c in charts]
    depth_vetoes = 0
    proposals = defaultdict(lambda: defaultdict(list))
    for i, boundary in enumerate(edges):
        for first, last, a, b, direction, length, bins, sample in boundary:
            x,y = [floor(a/4) for a in first]
            for dx,dy in product((-1,0,1),repeat=2):
                for j, start, end, c, d, other_length, other_bins, other_sample in by_edge[(direction,x+dx,y+dy)]:
                    if i==j or (bins is not None and other_bins is not None and not (bins & other_bins)):
                        continue
                    n1,n2=charts[j]['normal_axis'],charts[i]['normal_axis']
                    if abs(n1)==abs(n2):
                        continue
                    low = min(length, other_length)
                    start_px=float(np.linalg.norm(first-start))
                    end_px=float(np.linalg.norm(last-end))
                    if max(start_px,end_px) > min(edge_pixels, .19*low):
                        continue
                    if _distant_depth_edges(sample,other_sample):
                        depth_vetoes += 1
                        continue
                    shift = tuple(int(x) for x in c-a)
                    proposals[j,i][shift].append(dict(pixel_error=max(start_px,end_px),
                        xy=[start.tolist(),end.tolist()], a=tuple(c.tolist()), b=tuple(d.tolist())))
            by_edge[(direction,x,y)].append((i,first,last,a,b,length,bins,sample))
    result=[]
    for (i,j), options in proposals.items():
        rows=[]
        for shift, measurements in options.items():
            distinct = {(r['a'],r['b']):r for r in measurements}
            if len(distinct)>=min_edges:
                error=float(np.mean([r['pixel_error'] for r in distinct.values()]))
                rows.append(dict(i=i,j=j,shift=shift,edges=len(distinct),error=error,
                                image_edges=[r['xy'] for r in distinct.values()],kind='edge'))
        if not rows:
            continue
        rows.sort(key=lambda row:(-row['edges'],row['error'],row['shift']))
        if (len(rows)>1 and rows[0]['edges']==rows[1]['edges']
                and rows[1]['error']-rows[0]['error']<.40):
            continue
        result.append(rows[0])
    return sorted(result,key=lambda row:(-row['edges'],row['error'],row['i'],row['j'])), depth_vetoes


def _tangent_coordinates(chart):
    remaining = [axis for axis in range(3) if axis!=abs(chart['normal_axis'])-1]
    local_to_shared = np.array([axis_vector(c)[remaining] for c in chart['axes']]).T
    shared_to_local = np.eye(3)
    shared_to_local[:2,:2] = local_to_shared.T
    return np.asarray(chart['H'], float) @ shared_to_local, local_to_shared, remaining


def _samples(chart, matrix):
    cells = np.array([[row['u']+.5,row['v']+.5] for row in chart['cells']])
    # Separated visible cell centers independently verify that the relative scale
    # and phase agree over the chart, even if the two masks have a visible gap.
    candidates = {int(np.argmax(cells @ direction)) for direction in
                  ([1,1],[1,-1],[-1,1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1])}
    return cells[sorted(candidates)] @ matrix.T


def _evaluate_coplanar(a, b, pixel_tol):
    ha, ma, tangent = _tangent_coordinates(a)
    hb, mb, _ = _tangent_coordinates(b)
    pa, pb = _samples(a, ma), _samples(b, mb)
    if min(len(pa),len(pb)) < 3:
        return None
    if min(np.linalg.matrix_rank(p - p.mean(0)) for p in (pa,pb)) < 2:
        return None
    try:
        to_a = np.linalg.solve(ha,hb)
    except np.linalg.LinAlgError:
        return None
    mapped = [project(to_a,x) for x in pb]
    if any(x is None for x in mapped):
        return None
    shift = np.rint(np.median(np.asarray(mapped)-pb,axis=0)).astype(int)
    if np.max(abs(np.asarray(mapped)-pb-shift)) > .13:
        return None
    comparisons = [(p+shift,p) for p in pb] + [(p,p-shift) for p in pa]
    errors=[]
    for a_uv,b_uv in comparisons:
        x,y = project(ha,a_uv),project(hb,b_uv)
        if x is None or y is None:
            return None
        errors.append(float(np.linalg.norm(x-y)))
    if max(errors) > pixel_tol:
        return None
    d = np.zeros(3,dtype=int)
    d[tangent] = shift
    return tuple(d.tolist()), float(np.mean(errors)), len(errors)


def coplanar_joins(charts, pixel_tol):
    result=[]
    for i,j in combinations(range(len(charts)),2):
        a,b=charts[i],charts[j]
        if a['normal_axis']!=b['normal_axis']:
            continue
        fit=_evaluate_coplanar(a,b,pixel_tol)
        if fit is not None:
            shift,error,samples=fit
            result.append(dict(i=i,j=j,shift=shift,edges=0,error=error,independent_samples=samples,
                               image_edges=[],kind='coplanar'))
    return sorted(result,key=lambda row:(row['error'],-row['independent_samples'],row['i'],row['j']))


def solve_origins(count, proposals):
    """Accept consistent constraints and report contradictory joins."""
    connected=defaultdict(list)

    def component(anchor):
        origins={anchor:np.zeros(3,dtype=int)}
        queue=deque([anchor])
        while queue:
            here=queue.popleft()
            for there, shift in connected[here]:
                if there not in origins:
                    origins[there]=origins[here]+shift
                    queue.append(there)
        return origins

    used,rejected=[],[]
    for case in proposals:
        i,j=case['i'],case['j']
        shift=np.array(case['shift'],dtype=int)
        existing=component(i).get(j)
        if existing is not None and np.any(existing!=shift):
            rejected.append(case)
        else:
            connected[i].append((j,shift))
            connected[j].append((i,-shift))
            used.append(case)
    components=[]
    unseen=set(range(count))
    while unseen:
        origins=component(min(unseen))
        unseen.difference_update(origins)
        components.append(origins)
    return components,used,rejected


def assemble(charts, *, min_edges=2, edge_pixels=3.5, coplanar_pixels=None, relative_depth=None):
    charts,orientation=oriented(charts)
    proposals,depth_vetoes=propose_joins(charts,min_edges,relative_depth,edge_pixels)
    if coplanar_pixels is not None:
        proposals.extend(coplanar_joins(charts,coplanar_pixels))
    components,joins,conflicts=solve_origins(len(charts),proposals)
    groups=[]
    for origins in components:
        original_positions=set()
        blocks={}
        assigned=set()
        for k,t in sorted(origins.items(),key=lambda row: charts[row[0]].get('confidence',0.),reverse=True):
            c=charts[k]
            u,v=[axis_vector(a) for a in c['axes']]
            plane_normal=axis_vector(c['normal_axis'])
            offset=np.minimum(0,u)+np.minimum(0,v)-np.maximum(0,plane_normal)
            export=c.get('export_uv')
            allowed=None if export is None else set(map(tuple, export))
            for cell in c['cells']:
                xyz=tuple((t+offset+cell['u']*u+cell['v']*v).tolist())
                original_positions.add(xyz)
                if allowed is not None and (cell['u'],cell['v']) not in allowed:
                    continue
                if xyz not in blocks:
                    blocks[xyz]=cell['block_id']
                    if not cell.get('fallback'):
                        assigned.add(xyz)
        groups.append(dict(origins={charts[i]['id']:t.tolist() for i,t in origins.items()},
                           blocks=blocks, assigned=assigned, original_size=len(original_positions)))
    groups.sort(key=lambda c:(-c['original_size'],-len(c['origins'])))
    for edge in (*joins,*conflicts):
        edge['chart_ids']=[charts[edge[k]]['id'] for k in ('i','j')]
    return dict(groups=groups,joins=joins,conflicts=conflicts,
                oriented_charts=len(charts),orientation_hint=orientation,depth_vetoes=depth_vetoes)
