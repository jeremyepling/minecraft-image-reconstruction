"""Transfer a measured wall row lattice across directly observed vertical mask contacts."""
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark import relative_face_round as ordinary
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.full_block_lattice import atlas, verify_sprite
from engine.geometry import UNIT, project, transform
from engine.surface_model import lens

OUTPUT = ROOT / 'runs/shared-corner-rows-01'
HANDOFF = WORKTREES / 'floating-build-local-grid/runs/floating-build-grid-01/sky'
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
    a, b = sorted((a, b), key=lambda p: (p['rows'] is None, -p['support_area']))
    n = a['rows']
    result = dict(donor=a['label'], receiver=b['label'], donor_id=a['id'], receiver_id=b['id'],
                  donor_rows=n, receiver_rows=None, endpoints=edge.tolist(), decision='rejected',
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
    if b['rows'] is not None:
        return reject('receiver already has its own measured rows')
    if a['width'] is None or a['q'] is None or b['q'] is None:
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
    width = a['width']
    # Keep the original donor grid and put its corner at u=0 on both charts.
    H = transform(a['q']) @ np.array([[1/width,0,sides[0]],[0,-1/n,1.],[0,0,1.]])
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
    tangent = np.array(a['normal_camera']) * np.linalg.norm(np.linalg.solve(K, H[:,1]))
    R = H.copy()
    R[:,0] = K @ tangent
    if project(R, [[1,0]])[0,0] < project(R, [[0,0]])[0,0]:
        R[:,0] *= -1
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
    result.update(decision='accepted', reason='observed contact with a one-sided row period',
                  receiver_rows=vhi-vlo)
    return result, [donor, receiver]


def predict(case):
    name = case['id']
    folder = OUTPUT/'sky'/name
    handoff = HANDOFF/name/'source-handoff'
    packet = read(DENSE/name/'dense-charts.json')
    source = Path(case['variants']['sky']['image'])
    if not (digest(source) == case['variants']['sky']['source_sha256'] == packet['source_sha256']
            == read(handoff/'build-manifest.json')['image_sha256']):
        raise ValueError('Inconsistent source provenance')
    if any(digest(handoff/f'{k}.json') != packet[f'{k}_sha256'] for k in ('counts','surfaces')):
        raise ValueError('Cached handoffs disagree')
    image = Image.open(source).convert('RGB')
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
                          rows=confident_count(row,1), width=confident_count(row,0), mask=mask,
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
    save(folder/'seams.json',dict(id=name,source_sha256=packet['source_sha256'],seams=seams))
    charts = [c for c in packet['charts'] if c['id'] not in replacement] + list(replacement.values())
    base = dict(packet, baseline_chart_sha256=digest(DENSE/name/'dense-charts.json'))
    save(folder/'charts.json',dict(base, charts=charts))
    # Isolate the contribution of a transferred receiver from rasterizing the
    # measured donor itself. Preserve every other baseline chart and join rule.
    donors = {r['donor_id'] for r in seams if r['decision']=='accepted'}
    control = [c for c in packet['charts'] if c['id'] not in donors]
    control.extend(c for k,c in replacement.items() if k in donors)
    save(OUTPUT/'donor-only'/'sky'/name/'charts.json',dict(base, charts=control))
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
