"""Transfer a measured wall row lattice across directly observed vertical mask contacts."""
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark import relative_face_round as ordinary
from benchmark.expanded_baseline import WORKTREES, digest, read, save
from benchmark.floating_cloud_replay import SHARED_ROWS, handoff_provenance, source_handoff
from benchmark.full_block_lattice import atlas, verify_sprite
from engine.geometry import UNIT, project, transform
from engine.surface_model import lens
from engine.surface_perimeter import _native_alpha

OUTPUT = SHARED_ROWS
FROZEN_HANDOFF = WORKTREES / 'floating-build-local-grid/runs/floating-build-grid-01/sky'
DENSE = ordinary.ATLAS / 'sky'
# Rows must have an independently measured repeated texture period. A second
# width without a repeat measurement is emitted only in the first two columns.
MAX_RECEIVER_WIDTH = 2


def unpack(plane):
    width, height = plane['support']['size']
    mask = np.zeros((height, width), 'uint8')
    for y, left, right in plane['support']['runs']:
        mask[y, left:right] = 1
    return mask


def sample(mask, xy, scale):
    idx = np.floor(xy / scale).astype(int)
    x, y = idx[..., 0], idx[..., 1]
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    return valid & mask[y.clip(0, mask.shape[0]-1), x.clip(0, mask.shape[1]-1)].astype(bool)


def confident_count(row, axis):
    counts = row.get('counts', [])
    axes = row.get('evidence', {}).get('axes', [])
    if len(axes) != 2 or len(counts) != 2 or not counts[axis]:
        return None
    evidence = axes[axis] or {}
    n = round(counts[axis])
    if (evidence.get('source') == 'repeated texture' and evidence.get('agreement', 0) >= .45
            and evidence.get('rows', 0) >= 8 and 2 <= n <= 64
            and abs(counts[axis] - n) <= .35):
        return n
    return None


def row_strength(row):
    if confident_count(row, 1) is None:
        return 0.
    evidence = row['evidence']['axes'][1]
    return float(evidence['agreement']) * min(1., (evidence['rows'] / 24) ** .5)


def compare_rows(a, b, sides, heights):
    """Compare independent row grids at the same five native image heights."""
    measured = []
    for p, index in zip((a,b), sides):
        boundary = side(p['q'], index)
        xy = np.column_stack((x_at(boundary, heights), heights))
        uv = project(np.linalg.inv(transform(p['q'])), xy)
        measured.append((1-uv[:,1]) * p['rows'])
    delta = measured[0]-measured[1]
    offset = int(round(np.median(delta)))
    phase_error = float(np.max(np.abs(delta-offset)))
    span_error = float(np.ptp(delta))
    report = dict(donor_shared_span=float(np.ptp(measured[0])),
                  receiver_shared_span=float(np.ptp(measured[1])), integer_offset=offset,
                  phase_error_rows=phase_error, drift_rows=span_error,
                  donor_strength=a['strength'], receiver_strength=b['strength'])
    if phase_error <= .35 and span_error <= .4:
        return 'confirmed', report
    if a['strength'] >= .65 and a['strength'] - b['strength'] >= .15:
        return 'stronger-donor', report
    return 'ambiguous', report


def rotate_across(H, normal, K):
    """Use the source wall normal for the orthogonal one-block tangent."""
    tangent = np.array(normal) * np.linalg.norm(np.linalg.solve(K, H[:,1]))
    rotated = H.copy()
    rotated[:,0] = K @ tangent
    if project(rotated, [[1,0]])[0,0] < project(rotated, [[0,0]])[0,0]:
        rotated[:,0] *= -1
    return rotated


def contact(a, b, scale):
    touching = a['mask'] & b['near']
    y, x = np.nonzero(touching)
    if len(x) < 4:
        return None
    # Keep separate runs when an occluding surface changes edge at a ledge.
    lines = []
    for yy in np.unique(y):
        xx = x[y == yy]
        point = (np.median(xx) + .5, yy + .5)
        if not lines or point[1]-lines[-1][-1][1] > 3 or abs(point[0]-lines[-1][-1][0]) > 4:
            lines.append([])
        lines[-1].append(point)
    points = max(lines, key=len)
    if len(points) < 5:
        return None
    pts = np.array(points) * scale
    slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], 1)
    ys = pts[[0, -1], 1]
    return np.column_stack((slope * ys + intercept, ys))


def side(q, index):
    return q[[0, 3]] if index == 0 else q[[1, 2]]


def x_at(edge, ys):
    return edge[0, 0] + (ys-edge[0, 1]) * (edge[1, 0]-edge[0, 0]) / (edge[1, 1]-edge[0, 1])


def chart_cells(part, H, us, vs, rgb, scale):
    mask = part['mask']
    fractions = (np.arange(7)+.5)/7
    samples = np.stack(np.meshgrid(fractions, fractions), -1)
    cells = []
    for u in us:
        for v in vs:
            q = project(H, UNIT + (u, v))
            if np.linalg.norm(np.roll(q, 1, axis=0)-q, axis=1).min() < 3.5:
                continue
            support = sample(mask, project(H, samples + (u, v)), scale)
            if not support[3, 3] or support.mean() < .78:
                continue
            hit = verify_sprite(rgb, q[[3,2,1,0]])
            cells.append(dict(u=int(u), v=int(v), block_id=hit[0] if hit else 'smooth_stone',
                              confidence=hit[2] if hit else 0, fallback=hit is None,
                              mask_coverage=float(support.mean())))
    return dict(id=part['id'], normal_camera=part['normal_camera'], H=H.tolist(), cells=cells,
                confidence=float(np.mean([r['confidence'] for r in cells])) if cells else 0.,
                method='shared-corner-rows')


def pair(a, b, rgb, scale, K):
    edge = contact(a, b, scale)
    if edge is None:
        return
    # One measured side is enough; choose it before checking any scores.
    a, b = sorted((a, b), key=lambda p: (p['rows'] is None, -p['strength'], -p['support_area']))
    n = a['rows']
    result = dict(donor=a['label'], receiver=b['label'], donor_id=a['id'], receiver_id=b['id'],
                  donor_rows=n, receiver_rows=None, receiver_measured_rows=b['rows'],
                  endpoints=edge.tolist(), decision='rejected',
                  geometry={})
    def reject(reason):
        result['reason'] = reason
        return result, []
    if abs(np.dot(a['normal_camera'], b['normal_camera'])) > .28:
        return reject('wall normals do not form a right angle')
    d1, d2 = (p.get('relative_inverse_depth') for p in (a,b))
    result['geometry']['relative_inverse_depth'] = [d1, d2]
    if d1 and d2 and abs(d1-d2)/max(d1,d2) > .17:
        return reject('source depth changes across the contact')
    if n is None:
        return reject('neither side has a confident vertical repeat count')
    if a['q'] is None or b['q'] is None or (a['width'] is None and b['rows'] is None):
        return reject('missing measured donor width or quadrilateral')
    sides = []
    for p in (a, b):
        options = []
        for index in (0, 1):
            s = side(p['q'], index)
            if s[1,1] <= s[0,1]+5:
                continue
            error = abs(x_at(s, edge[:,1])-edge[:,0]).max()
            options.append((error, index))
        if not options or min(options)[0] > 6:
            return reject('contact does not follow both chart boundaries')
        sides.append(min(options)[1])
    if sides[0] == sides[1]:
        return reject('chart interiors lie on the same side')
    left, right = (side(p['q'], i) for p, i in zip((a,b), sides))
    top = max(left[0,1], right[0,1], edge[0,1]-scale[1])
    bottom = min(left[1,1], right[1,1], edge[1,1]+scale[1])
    if bottom <= top:
        return reject('no boundary overlap')
    reason = 'observed contact with a one-sided row period'
    if b['rows'] is not None:
        outcome, evidence = compare_rows(a, b, sides, np.linspace(top, bottom, 5))
        result['row_comparison'] = dict(outcome=outcome, **evidence)
        if outcome == 'ambiguous':
            return reject('measured row grids disagree without a reliable winner')
        reason = ('both measured row grids agree on the shared segment' if outcome == 'confirmed'
                  else 'stronger measured row grid overrides the conflicting receiver on the shared segment')
    width = a['width'] or MAX_RECEIVER_WIDTH
    # Keep the original donor grid and put its corner at u=0 on both charts.
    H = transform(a['q']) @ np.array([[1/width,0,sides[0]],[0,-1/n,1.],[0,0,1.]])
    if a['width'] is None:
        H = rotate_across(H, b['normal_camera'], K)
    ends = np.column_stack((x_at(left, np.array([top,bottom])), [top,bottom]))
    result['endpoints'] = ends.tolist()
    vtop, vbottom = project(np.linalg.inv(H), ends)[:,1]
    vlo = max(0, int(np.ceil(vbottom-.15)))
    vhi = min(n, int(np.floor(vtop+.15)))
    if vhi-vlo < 2:
        return reject('shared contact shorter than two complete rows')
    # Rotate the measured unit tangent about the vertical seam. Its new
    # direction is the donor wall normal; its magnitude is one donor row
    # expressed in homogeneous camera coordinates at the assumed source lens.
    R = rotate_across(H, a['normal_camera'], K)
    us = range(-width,0) if sides[0] else range(width)
    rs = range(-MAX_RECEIVER_WIDTH,0) if sides[1] else range(MAX_RECEIVER_WIDTH)
    donor = chart_cells(a, H, us, range(n), rgb, scale)
    receiver = chart_cells(b, R, rs, range(vlo,vhi), rgb, scale)
    donor_edge = -1 if sides[0] else 0
    receiver_edge = -1 if sides[1] else 0
    shared = sorted({c['v'] for c in donor['cells'] if c['u']==donor_edge} &
                    {c['v'] for c in receiver['cells'] if c['u']==receiver_edge})
    result['geometry'].update(donor_side=sides[0], receiver_side=sides[1], donor_width=width,
                              receiver_column_cap=MAX_RECEIVER_WIDTH, donor_v_interval=[vlo,vhi],
                              shared_supported_row_ids=shared, donor_cells=len(donor['cells']),
                              receiver_cells=len(receiver['cells']),
                              receiver_supported_row_ids=sorted({c['v'] for c in receiver['cells']}),
                              row_ticks=project(H, [[0,v] for v in range(vlo,vhi+1)]).tolist())
    if len(shared) < 2:
        return reject('fewer than two common row edges have supported cells')
    result.update(decision='accepted', reason=reason,
                  receiver_rows=vhi-vlo)
    return result, [donor, receiver]


def dense_reuse(packet, frozen, handoff, image):
    """Check masks, consumed count fields, and shared direction before reusing charts."""
    from benchmark.camera_structure_scaffold import outline_lines
    from benchmark.relative_dense_charts import vertical_pencil
    from benchmark.relative_source_charts import wall_quad

    def load_inputs(folder):
        planes = {p['id']: p for p in read(folder/'surfaces.json')['planes']}
        rows = {r['id']: r for r in read(folder/'counts.json')['surfaces']}
        return planes, rows

    old_planes, old_rows = load_inputs(frozen)
    planes, rows = load_inputs(handoff)
    for sid, plane in planes.items():
        old = old_planes.get(sid, {})
        if any(plane.get(k) != old.get(k) for k in ('support', 'normal_camera', 'relative_inverse_depth')):
            return None
        row, old_row = rows.get(sid, {}), old_rows.get(sid, {})
        if (any(row.get(k) != old_row.get(k) for k in ('quad', 'counts'))
                or row.get('evidence', {}).get('aspect') != old_row.get('evidence', {}).get('aspect')):
            return None

    def pencil(parts, count_rows):
        walls = [p for sid, p in parts.items() if wall_quad(count_rows.get(sid, {}), p) is not None]
        lines, _ = outline_lines(image, walls, [])
        return vertical_pencil(lines, image.size)

    if not np.array_equal(pencil(old_planes, old_rows), pencil(planes, rows)):
        return None
    removed = np.zeros(image.size[::-1], bool)
    removed_ids = old_planes.keys() - planes.keys()
    for sid in removed_ids:
        alpha, (left, top, right, bottom) = _native_alpha(old_planes[sid]['support'], image.size)
        removed[top:bottom, left:right] |= np.asarray(alpha) != 0
    y, x = np.nonzero(removed)
    pixels = np.column_stack((x + .5, y + .5))
    for chart in packet['charts']:
        if chart['id'] not in planes:
            return None
        uv = project(np.linalg.inv(chart['H']), pixels)
        for cell in chart['cells']:
            inside = ((uv[:, 0] >= cell['u']) & (uv[:, 0] < cell['u'] + 1)
                      & (uv[:, 1] >= cell['v']) & (uv[:, 1] < cell['v'] + 1))
            if inside.any():
                return None
    return dict(removed_ids=sorted(removed_ids), removed_pixels=len(pixels),
                unchanged_vertical_pencil=True, overlapping_pixels=0,
                checked_cells=sum(len(c['cells']) for c in packet['charts']))


def dense_packet(case, handoff, image, provenance):
    from benchmark.relative_source_charts import extract

    name = case['id']
    spec = case['variants']['sky']
    path = DENSE/name/'dense-charts.json'
    frozen = FROZEN_HANDOFF/name/'source-handoff'
    packet = read(path)
    if not (spec['source_sha256'] == packet['source_sha256'] == read(frozen/'build-manifest.json')['image_sha256']
            and atlas().provenance['atlas_sha256'] == packet['atlas_sha256']
            and all(digest(frozen/f'{key}.json') == packet[f'{key}_sha256'] for key in ('counts', 'surfaces'))):
        raise ValueError(f'Frozen dense charts failed provenance check: {path}')
    proof = dense_reuse(packet, frozen, handoff, image)
    if proof is not None:
        return packet | provenance | dict(
            baseline_chart_sha256=digest(path), dense_chart_reuse=proof,
            frozen_surfaces_sha256=packet['surfaces_sha256'], frozen_counts_sha256=packet['counts_sha256'])
    regenerated = OUTPUT/'dense-source'/'sky'/name
    extract((spec, name, 'sky', handoff, regenerated))
    path = regenerated/'dense-charts.json'
    return read(path) | provenance | dict(baseline_chart_sha256=digest(path), dense_chart_reuse=None)


def predict(case):
    name = case['id']
    folder = OUTPUT/'sky'/name
    handoff = source_handoff(name)
    source = Path(case['variants']['sky']['image'])
    sha = digest(source)
    if sha != case['variants']['sky']['source_sha256']:
        raise ValueError('Inconsistent source provenance')
    provenance = handoff_provenance(handoff, sha)
    image = Image.open(source).convert('RGB')
    packet = dense_packet(case, handoff, image, provenance)
    rgb = np.asarray(image)
    rows = {r['id']:r for r in read(handoff/'counts.json')['surfaces']}
    planes = sorted(read(handoff/'surfaces.json')['planes'], key=lambda p:-p['support_area'])
    parts = []
    for i, p in enumerate(planes):
        if not p.get('normal_camera') or abs(p['normal_camera'][1]) > .63 or not p.get('support'):
            continue
        row = rows.get(p['id'], {})
        mask = unpack(p)
        parts.append(dict(p, label=f'S{i+1:02d}', q=np.array(row['quad']) if row.get('quad') else None,
                          rows=confident_count(row,1), strength=row_strength(row), width=confident_count(row,0), mask=mask,
                          near=cv2.dilate(mask,np.ones((5,5),'uint8'))))
    seams = []
    replacement = {}
    K,_ = lens([],image.size)
    for a,b in combinations(parts,2):
        scale = np.array(image.size)/a['support']['size']
        answer = pair(a,b,rgb,scale,K)
        if answer is None:
            continue
        record, charts = answer
        if charts and any(c['id'] in replacement for c in charts):
            record.update(decision='rejected',reason='wall already participates in a stronger area-ranked contact')
        else:
            replacement.update({c['id']:c for c in charts})
        seams.append(record)
    save(folder/'seams.json',dict(id=name,source_sha256=sha,seams=seams,**provenance))
    charts = [c for c in packet['charts'] if c['id'] not in replacement] + list(replacement.values())
    save(folder/'charts.json',dict(packet, charts=charts))
    # Isolate the contribution of a transferred receiver from rasterizing the
    # measured donor itself. Preserve every other baseline chart and join rule.
    donors = {r['donor_id'] for r in seams if r['decision']=='accepted'}
    control = [c for c in packet['charts'] if c['id'] not in donors]
    control.extend(c for k,c in replacement.items() if k in donors)
    save(OUTPUT/'donor-only'/'sky'/name/'charts.json',dict(packet, charts=control))
    print(name, [(r['donor'],r['receiver'],r['donor_rows'],r['receiver_rows']) for r in seams if r['decision']=='accepted'], flush=True)


def main():
    if not atlas().provenance['atlas_sha256']:
        raise ValueError('Source atlas unavailable')
    cases = read(ordinary.SHARED/'fixtures.json')['cases']
    for case in cases:
        predict(case)
    # Register a private chart source with the frozen assembly/evaluation entry point.
    mode = OUTPUT.name
    ordinary.OUTPUT = OUTPUT.parent
    ordinary.MODES[mode] = (OUTPUT, 'charts.json', False, 1, 2.5, .5, False)
    for case in cases:
        print(ordinary.evaluate((case,'sky',mode)), flush=True)
    ordinary.OUTPUT = OUTPUT
    ordinary.MODES['donor-only'] = (OUTPUT/'donor-only','charts.json',False,1,2.5,.5,False)
    for case in cases:
        print(ordinary.evaluate((case,'sky','donor-only')), flush=True)


if __name__ == '__main__':
    main()
