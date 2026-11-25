"""Lift visible rectangular outlines into camera coordinates (+x right, +y down, +z forward)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine.geometry import UNIT, transform


DEFAULTS = Path(__file__).with_name('surface_defaults.json')
SETTINGS = dict(assumed_vertical_fov_degrees=70, shared_edge_tolerance_px=5,
                minimum_shared_edge_px=10, max_relative_edge_disagreement=.20,
                max_pitch_anisotropy=1.25, min_fold_degrees=12)


def intrinsic(size, focal):
    w, h = size
    return np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.]])


def lens(quads, size):
    """Perpendicular directions supply a focal length for centered square pixels."""
    w, h = size
    centered = intrinsic(size, 1)
    estimates = []
    for q in quads:
        try:
            a, b = (np.linalg.solve(centered, transform(q))[:, :2]).T
        except np.linalg.LinAlgError:
            continue
        if abs(a[2] * b[2]) < 1e-10:
            continue
        squared = -a[:2] @ b[:2] / (a[2] * b[2])
        if (.35*h)**2 < squared < (3*h)**2:
            estimates.append(float(np.sqrt(squared)))
    if estimates:
        return intrinsic(size, np.median(estimates)), 'orthogonal-outline-vanishing-points'
    f = h / (2 * np.tan(np.radians(SETTINGS['assumed_vertical_fov_degrees']) / 2))
    return intrinsic(size, f), 'assumed-70-degree-vertical-fov'


def pitch(image, quad, version, method):
    if method == 'repeat':
        from engine.surface_pitch import fit
        return fit(image, quad, version)
    if method != 'texel':
        raise ValueError(f'Unknown pitch method {method}')
    from engine.texel_grid import texel_grid
    counts = list(map(int, texel_grid(np.asarray(image), quad)))
    grid = dict(phase_blocks=[0, 0], grid_to_image=(transform(quad) @
                np.diag([1/counts[0], 1/counts[1], 1])).tolist())
    return dict(counts=counts, grid=grid, confidence=None,
                reason='Legacy texel spacing; first chart corner assumed to lie on a seam.')


def canonical(quad):
    """Orient the two chart coordinates predominantly right and down in the image."""
    q = np.asarray(quad, float)
    choices = [np.roll(a, n, axis=0) for a in (q, q[::-1]) for n in range(4)]
    def alignment(p):
        u, v = p[1] + p[2] - p[0] - p[3], p[3] + p[2] - p[0] - p[1]
        return u[0]/max(1e-9, np.linalg.norm(u)) + v[1]/max(1e-9, np.linalg.norm(v))
    return max(choices, key=alignment).tolist()


def lift(q, K, result):
    try:
        ray = np.linalg.solve(K, transform(q))
    except np.linalg.LinAlgError:
        return None
    xyz = np.column_stack((UNIT, np.ones(4))) @ ray.T
    if np.min(xyz[:, 2]) <= .01:
        return None
    normal = np.cross(ray[:, 0], ray[:, 1])
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    normal *= -np.sign(normal @ xyz[0]) / norm
    counts = result.get('counts')
    scale = None
    if counts is not None:
        scales = counts / np.linalg.norm(ray[:, :2], axis=0)
        if np.isfinite(scales).all() and min(scales) > 0 and max(scales)/min(scales) <= SETTINGS['max_pitch_anisotropy']:
            scale = float(np.mean(scales) * xyz[:, 2].mean())
    return xyz / xyz[:, 2].mean(), normal, scale


def shared_edge(one, two):
    """Only stitch two complete, nearly coincident edges of differently oriented planes."""
    q, xyz, normal = one
    r, pqr, other = two
    if abs(normal @ other) > np.cos(np.radians(SETTINGS['min_fold_degrees'])):
        return None
    best = None
    for a in range(4):
        ai = [a, (a+1)%4]
        if np.linalg.norm(q[ai[1]] - q[a]) < SETTINGS['minimum_shared_edge_px']:
            continue
        for b in range(4):
            for bi in ([b, (b+1)%4], [(b+1)%4, b]):
                miss = np.linalg.norm(q[ai] - r[bi], axis=1).max()
                if miss > SETTINGS['shared_edge_tolerance_px'] or (best and miss >= best[0]):
                    continue
                ratios = xyz[ai, 2] / pqr[bi, 2]
                if max(ratios)/min(ratios) <= 1 + SETTINGS['max_relative_edge_disagreement']:
                    best = float(miss), float(np.log(ratios).mean()), ai, bi
    return best


def assemble(proposals, pitches, size, diagnostic_K=None):
    """Solve connected components' relative depth from coincident edge endpoints."""
    if diagnostic_K is None:
        K, source = lens([p['quad'] for p in proposals], size)
    else:
        K, source = diagnostic_K, 'recorded-camera-diagnostic'
    planes, items = [], {}
    for p, result in zip(proposals, pitches, strict=True):
        row = dict(p, pitch=result)
        lifted = lift(p['quad'], K, result)
        if lifted:
            xyz, normal, scale = lifted
            i = len(planes)
            items[i] = np.asarray(p['quad']), xyz, normal, scale
            row.update(normal_camera=normal.tolist())
        planes.append(row)
    edges, graph = [], {i: [] for i in items}
    for i in items:
        for j in items:
            if j <= i:
                continue
            match = shared_edge(items[i][:3], items[j][:3])
            if match is None:
                continue
            miss, log_ratio, ai, bi = match
            direct_i, direct_j = items[i][3], items[j][3]
            if direct_i and direct_j and abs(np.log(direct_j/direct_i)-log_ratio) > np.log(1.25):
                continue
            graph[i].append(j)
            graph[j].append(i)
            edges.append((i, j, log_ratio, dict(planes=[planes[i]['id'], planes[j]['id']],
                         corners=[ai, bi], endpoint_max_px=miss)))
    todo = set(items)
    component = 0
    while todo:
        root = min(todo)
        stack, group = [root], []
        while stack:
            i = stack.pop()
            if i in todo:
                todo.remove(i)
                group.append(i)
                stack.extend(graph[i])
        index = {i: k for k, i in enumerate(group)}
        equations, target = [], []
        def constraint(weights, value):
            row = np.zeros(len(group))
            for i, weight in weights.items():
                row[index[i]] = weight
            equations.append(row)
            target.append(value)
        for i, j, ratio, _ in edges:
            if i in index:
                constraint({i: -1, j: 1}, ratio)
        anchors = [i for i in group if items[i][3] is not None]
        for i in anchors:
            constraint({i: 1}, np.log(items[i][3]))
        if not anchors:
            constraint({root: 1}, 0.)
        scales = np.exp(np.linalg.lstsq(equations, target, rcond=None)[0])
        for i in group:
            planes[i].update(component=component, units='blocks' if anchors else 'relative',
                             scale_source=('local-pitch' if i in anchors else 'shared-edge') if anchors else 'unknown',
                             corners_camera=(items[i][1] * scales[index[i]]).tolist())
        component += 1
    return dict(lens_source=source, intrinsics=K.tolist(), planes=planes,
                shared_edges=[edge[-1] for edge in edges])


def proposals_for(image, version, method, normals=None):
    if method == 'contact':
        from engine.camera_grid import predict
        parts = predict(image, version, 'seam-fit')['planes']
        return [dict(id=f'contact-{i}', quad=canonical(p['quad'])) for i,p in enumerate(parts)]
    if method in ('normal-rank','normal-regions'):
        if normals is None:
            from engine.surface_fields import infer
            normals=infer(image,'normal')
        if method=='normal-rank':
            from engine.normal_rank import detect
        else:
            from engine.normal_regions import detect
        parts=detect(image,normals,version=version)
        from engine.surface_fields import observed_normal
        for p in parts:
            p['observed_normal_camera']=observed_normal(p['quad'],normals,image.size)
    elif method in ('edges','edges+shade'):
        from engine.surface_edges import detect_planes
        parts=detect_planes(image,version=version,shading=method=='edges+shade')
    else:
        raise ValueError(f'Unknown outline method {method}')
    reordered = {}
    for p in parts:
        old = p['quad']
        p['quad'] = canonical(old)
        reordered[p['id']] = [p['quad'].index(corner) for corner in old]
    for p in parts:
        for link in p.get('corner_junctions',[]):
            link['corner'] = reordered[p['id']][link['corner']]
            link['neighbor_corner'] = reordered[link['neighbor']][link['neighbor_corner']]
    return parts


def build(image_path, version, out):
    from engine.atlas import digest
    if version != '26.3':
        raise ValueError('The surface methods support Java 26.3.')
    methods = json.loads(DEFAULTS.read_text())
    with Image.open(image_path) as src:
        image = src.convert('RGB')
    patches = proposals_for(image, version, methods['outlines'])
    model = assemble(patches,[pitch(image,p['quad'],version,methods['pitch']) for p in patches],image.size)
    if 'order' in methods:
        from engine.surface_order import order
        field=None
        if methods['order']=='depth-anything':
            from engine.surface_fields import infer
            field=infer(image,'depth')
        model['relative_depth_order']=order(model['planes'],image.size,field)
    out.mkdir(parents=True,exist_ok=True)
    (out/'surfaces.json').write_text(json.dumps(model, allow_nan=False, indent=2))
    manifest = dict(version=version, through='surfaces', image_sha256=digest(image_path),
                    stages=dict(**methods, lens=model['lens_source'], lift='shared-edges-v1'),
                    settings=SETTINGS, artifact='surfaces.json')
    if methods['outlines'].startswith('normal-') or methods.get('order')=='depth-anything':
        from engine.surface_fields import MODELS, SETTINGS as inference
        fields=(['normal'] if methods['outlines'].startswith('normal-') else [])+(['depth'] if methods.get('order')=='depth-anything' else [])
        manifest['model_sha256']={k:MODELS[k][2] for k in fields}
        manifest['inference_settings']=inference
    (out/'build-manifest.json').write_text(json.dumps(manifest, indent=2))
