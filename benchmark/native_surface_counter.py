"""Measure local block pitch and sprite phase directly against masked source pixels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


from benchmark import relative_face_round as ordinary
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.full_block_lattice import atlas, verify_sprite
from benchmark.shared_corner_rows import HANDOFF, unpack
from engine.atlas_count import _shortlist
from engine.geometry import UNIT, project
from engine.surface_model import lens

OUTPUT = ROOT / 'runs/surface-crop-first-01'
BASE = WORKTREES / 'shared-corner-rows/runs/shared-corner-rows-01/sky'
FRACTIONS = (np.arange(16) + .5) / 16
SAMPLES = np.stack(np.meshgrid(FRACTIONS, FRACTIONS), axis=-1)
MAX_POINTS = 1536


def remap(image, points):
    shape = points.shape[:-1]
    # OpenCV has a 32767-pixel limit per map dimension.
    grid = (points - .5).astype('f').reshape(-1, shape[-1], 2)
    result = cv2.remap(image, grid, None, cv2.INTER_LINEAR)
    return result.reshape(shape)


def basis(normal, up, origin, K):
    right = np.cross(normal, up)
    right /= np.linalg.norm(right)
    down = np.cross(normal, right)
    anchor = np.linalg.solve(K, [*origin, 1])
    raw = K @ np.column_stack((right, down, anchor))
    delta = project(raw, .001*np.eye(2)) - origin
    for i in (0, 1):
        if delta[i, i] < 0:
            raw[:, i] *= -1
    raw[:, :2] *= .001 / np.linalg.norm(delta[1])
    return raw


def edge_basis(rgb, mask, center):
    ys,xs = np.nonzero(mask)
    x,y = xs.min(),ys.min()
    crop = cv2.cvtColor(rgb[y:ys.max()+1,x:xs.max()+1],cv2.COLOR_RGB2GRAY)
    if min(crop.shape)<10:
        return None
    lines = cv2.createLineSegmentDetector().detect(crop)[0]
    if lines is None:
        return None
    ends = lines[:,0].reshape(-1,2,2) + [x+.5,y+.5]
    delta = ends[:,1]-ends[:,0]
    length = np.linalg.norm(delta,axis=1)
    mid = ends.mean(1)
    support = remap(mask.astype('f'), ends[:,0,None]+delta[:,None]*np.linspace(.15,.85,5)[None,:,None]).mean(1)>.85
    H = np.eye(3)
    H[:2,2] = center
    evidence = []
    for axis in (0,1):
        other = 1-axis
        keep = support & (length>=7) & (abs(delta[:,other])<.65*abs(delta[:,axis]))
        if keep.sum()<4:
            return None
        slopes = delta[keep,other]/delta[keep,axis]
        weights = np.minimum(length[keep],45)
        offset = mid[keep]-center
        # t_other + (center_other-mid_other)*c = slope * (1+(center_axis-mid_axis)*c)
        A = np.column_stack((np.ones(len(slopes)), slopes*offset[:,axis]-offset[:,other]))
        estimate = np.array([np.median(slopes),0.])
        for _ in range(5):
            w = np.sqrt(weights/np.maximum(.025,abs(A@estimate-slopes)))
            estimate = np.linalg.lstsq(w[:,None]*A, w*slopes,rcond=None)[0]
            estimate[1] = np.clip(estimate[1],-.002,.002)
        t = np.eye(2)[axis].copy()
        t[other] = estimate[0]
        c = estimate[1]
        H[:,axis] = [*(t+center*c),c]
        evidence.append(dict(segments=int(keep.sum()),slope=round(float(t[other]),4),
                             inverse_vanishing_distance=round(float(c),6)))
    return H,evidence


def pitch_scan(gray, valid, points, uv, H, axis):
    source = remap(gray, points)
    step = np.eye(2)[axis]
    response = []
    for lag in np.arange(3, 55.1, .5):
        ends = project(H, uv + lag*step)
        ok = remap(valid, ends) > .99
        if ok.sum() < max(48, .10*len(ok)):
            response.append(0.)
            continue
        a = source[ok]
        b = remap(gray, ends)[ok]
        a = a - a.mean()
        b = b - b.mean()
        response.append(float(a @ b / max(1., np.linalg.norm(a)*np.linalg.norm(b))))
    return np.asarray(response)


def candidates(responses, edge):
    peaks = []
    for response in responses:
        idx = [i for i in range(2,len(response)-2) if response[i]>response[i-1] and response[i]>=response[i+1] and response[i]-min(response[max(0,i-4):i])>.05 and response[i]-min(response[i+1:i+5])>.05]
        peaks.append([dict(pitch=3+.5*int(i), correlation=round(float(response[i]), 3))
                      for i in sorted(idx, key=lambda j: -response[j])[:6]])
    proposed = []
    for u in peaks[0]:
        for v in peaks[1]:
            a, b = u['pitch'], v['pitch']
            if (not edge and abs(a-b) > max(1.5, .14*min(a,b))) or not .28 <= a/b <= 1.8 or min(u['correlation'],v['correlation']) < .16:
                continue
            proposed.append((u['correlation']+v['correlation'], a,b))
    return peaks, sorted(proposed, reverse=True)[:6]


def search_phase(rgb, mask, H, seeds, tile_ids):
    cat = atlas()
    tiles = cat.patterns[tile_ids].reshape(-1, 256)
    gray = rgb.mean(2, dtype=np.float32)
    # Search a complete two-dimensional phase at each interior observation.
    phase = np.stack(np.meshgrid(np.arange(8)/8, np.arange(8)/8), axis=-1).reshape(-1,2)
    proposals = []
    for point in seeds:
        origins = point - phase
        xy = project(H, origins[:,None,None,:]+SAMPLES)
        valid = remap(mask, xy).reshape(-1,256).min(1) > .99
        if not valid.any():
            continue
        patch = remap(gray, xy).reshape(-1,256)
        patch -= patch.mean(1,keepdims=True)
        std = patch.std(1)
        quality = (patch @ tiles.T).max(1) / (256*np.maximum(std,1.))
        quality[(std<2) | ~valid] = -1
        best = int(quality.argmax())
        if quality[best] >= .43:
            proposals.append((float(quality[best]), origins[best]))
    return sorted(proposals, key=lambda item:-item[0])[:3]


def verified_grid(rgb, mask, H, uv, origin):
    H = H @ np.array([[1,0,origin[0]],[0,1,origin[1]],[0,0,1.]])
    local = uv-origin
    low = np.floor(local.min(0)).astype(int)
    high = np.ceil(local.max(0)).astype(int)
    if np.prod(high-low) > 1800:
        return []
    hits = []
    for v in range(low[1],high[1]):
        for u in range(low[0],high[0]):
            xy = project(H, SAMPLES[::3,::3]+(u,v))
            if remap(mask, xy).min() < .99:
                continue
            q = project(H, UNIT + (u,v))
            if np.linalg.norm(q-np.roll(q,1,axis=0),axis=1).min() < 4:
                continue
            hit = verify_sprite(rgb,q)
            if hit:
                hits.append(dict(u=int(u),v=int(v),block_id=hit[0],texture=hit[1],
                                 correlation=round(hit[2],4),quad=q.round(2).tolist()))
    return hits


def measure(rgb, gray, p, up, K, label):
    rec = dict(id=p['id'], label=label, decision='rejected', matches=[], pitch_candidates=[[],[]])
    def reject(reason):
        return dict(rec, reason=reason), None
    n = p.get('normal_camera')
    if n is None or abs(n[1]) >= .65 or up is None:
        return reject('wall directions unavailable')
    mask = cv2.resize(unpack(p), (rgb.shape[1],rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = cv2.erode(mask,np.ones((5,5),'uint8')).astype('f')
    ys, xs = np.nonzero(valid)
    if len(xs) < 128:
        return reject('too few interior source pixels')
    i = np.linspace(0,len(xs)-1,min(len(xs),MAX_POINTS)).astype(int)
    points = np.column_stack((xs[i],ys[i]))+.5
    center = np.median(points,axis=0)
    edges = edge_basis(rgb,mask,center)
    if edges:
        H,rec['edge_fit'] = edges
    else:
        H = basis(n,up,center,K)
    uv = project(np.linalg.inv(H),points)
    if not np.isfinite(uv).all() or np.max(abs(uv)) > 1e5:
        return reject('unstable source directions')
    response = [pitch_scan(gray,valid,points,uv,H,a) for a in (0,1)]
    peaks, proposed = candidates(response, bool(edges))
    rec.update(pitch_candidates=peaks, origin_px=center.tolist(),
               unit_vectors_px=(project(H,np.eye(2))-center).round(4).tolist(),
               joint_pitches=[list(r) for r in proposed])
    if not proposed:
        return reject('no compatible pair of independent native pitch peaks')
    tile_ids,_ = _shortlist(rgb[ys[i],xs[i]],np.ones(len(i),bool),atlas())
    dist = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    seeds = []
    work = dist.copy()
    for _ in range(9):
        _, value, _, (x,y) = cv2.minMaxLoc(work)
        if value < 4:
            break
        seeds.append([x+.5,y+.5])
        cv2.circle(work,(x,y),max(10,int(value)),0.,-1)
    if not seeds:
        return reject('no sufficiently wide native patch')
    best = None
    for _, a,b in proposed:
        P = H @ np.diag([a,b,1.])
        measure_uv = uv / [a,b]
        centers = project(np.linalg.inv(P),np.array(seeds))
        for phase_score, seed in search_phase(rgb,mask.astype('f'),P,centers,tile_ids):
            hits = verified_grid(rgb,mask.astype('f'),P,measure_uv,seed)
            pairs = {(r['u'],r['v']) for r in hits}
            neighbors = sum((u+1,v) in pairs or (u,v+1) in pairs for u,v in pairs)
            key = (neighbors,len(hits),phase_score)
            if best is None or key>best[0]:
                best = (key,hits,P,seed,a,b)
    if best is None:
        return reject('no complete local sprite phase matched')
    (neighbors,total,_), hits,P,seed,a,b = best
    chart = P @ np.array([[1,0,seed[0]],[0,1,seed[1]],[0,0,1.]])
    intervals = project(np.linalg.inv(chart),points)
    rec.update(matches=hits, pitch_at_origin_px=np.linalg.norm(project(P,np.eye(2))-center,axis=1).round(2).tolist(),
               span_blocks=np.ptp(intervals,axis=0).round(2).tolist(),
               matched_columns=sorted({r['u'] for r in hits}), matched_rows=sorted({r['v'] for r in hits}),
               adjacent_matches=neighbors)
    if total < 3 or neighbors < 2:
        return reject('fewer than three verified cells or two neighbor links')
    cells = [dict(u=r['u'],v=r['v'],block_id=r['block_id'],confidence=r['correlation'],fallback=False) for r in hits]
    chart_row = dict(id=p['id'],normal_camera=n,H=chart.tolist(),cells=cells,
                     confidence=float(np.mean([r['confidence'] for r in cells])),method='native-mask-sprite')
    return dict(rec,decision='accepted',reason='repeats on both axes and adjacent native atlas matches'), chart_row


def fill_fallback_ids(packet, surfaces):
    """Keep chart corners and cells fixed; transfer only a congruent, high-confidence ID."""
    charts = []
    transfers = []
    by_id = {row['id']:row['matches'] for row in surfaces if row['decision']=='accepted'}
    for chart in packet['charts']:
        cells = []
        for face in chart['cells']:
            cells.append(face)
            if not face.get('fallback') or chart['id'] not in by_id:
                continue
            quad = project(np.asarray(chart['H']),UNIT+[face['u'],face['v']]).astype('f')
            area = cv2.contourArea(quad)
            candidates = []
            for hit in by_id[chart['id']]:
                if hit['correlation']<.68:
                    continue
                source_quad = np.asarray(hit['quad'],'f')
                hit_area = cv2.contourArea(source_quad)
                intersection,_ = cv2.intersectConvexConvex(quad,source_quad)
                if min(area,hit_area)<=0 or intersection/max(area,hit_area)<.65:
                    continue
                iou = intersection/(area+hit_area-intersection)
                if iou>=.50:
                    candidates.append((iou,hit))
            if not candidates:
                continue
            iou,hit = max(candidates,key=lambda pair:pair[0])
            cells[-1] = dict(face,block_id=hit['block_id'],fallback=False,confidence=hit['correlation'])
            transfers.append(dict(surface=chart['id'],u=face['u'],v=face['v'],block_id=hit['block_id'],
                                  iou=round(iou,4),source_quad=hit['quad']))
        charts.append(dict(chart,cells=cells))
    return dict(packet,charts=charts), transfers


def predict(case):
    cv2.setNumThreads(1)
    folder = OUTPUT/'sky'/case['id']
    handoff = HANDOFF/case['id']/'source-handoff'
    baseline = BASE/case['id']/'charts.json'
    packet = read(baseline)
    source = Path(case['variants']['sky']['image'])
    sha = digest(source)
    if sha != packet['source_sha256'] or sha != read(handoff/'build-manifest.json')['image_sha256']:
        raise ValueError('Screenshot hashes disagree')
    rgb = np.array(Image.open(source).convert('RGB'))
    mono = rgb.mean(2,dtype=np.float32)
    gray = mono - cv2.GaussianBlur(mono,(0,0),2.)
    K,_ = lens([], (rgb.shape[1],rgb.shape[0]))
    planes = sorted(read(handoff/'surfaces.json')['planes'],key=lambda p:-p['support_area'])
    tops = [p['normal_camera'] for p in planes if p.get('normal_camera') and abs(p['normal_camera'][1])>=.68]
    up = np.array(tops[0]) if tops else None
    walls = [np.array(p['normal_camera']) for p in planes if p.get('normal_camera') and abs(p['normal_camera'][1])<.65]
    if up is None and walls:
        other = next((n for n in walls if abs(walls[0]@n)<.25),None)
        if other is not None:
            up = np.cross(walls[0],other)
            up /= np.linalg.norm(up)
    evidence, replacement = [], {}
    for idx,plane in enumerate(planes):
        r,chart = measure(rgb,gray,plane,up,K,f'S{idx+1:02d}')
        evidence.append(r)
        if chart:
            replacement[chart['id']] = chart
    save(folder/'native-evidence.json',dict(source_sha256=sha,atlas=atlas().provenance,
         surfaces_sha256=digest(handoff/'surfaces.json'),baseline_chart_sha256=digest(baseline),surfaces=evidence))
    charts = [c for c in packet['charts'] if c['id'] not in replacement]+list(replacement.values())
    save(folder/'charts.json',dict(packet,charts=charts))
    material, transfers = fill_fallback_ids(packet,evidence)
    save(OUTPUT/'id-only'/'sky'/case['id']/'charts.json',material)
    save(folder/'id-transfers.json', transfers)
    print(case['id'],list(replacement),flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--workers',type=int,default=3)
    cli.add_argument('--frame',default='')
    args = cli.parse_args()
    cases = [c for c in read(ordinary.SHARED/'fixtures.json')['cases'] if args.frame in c['id']]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(predict,cases))
