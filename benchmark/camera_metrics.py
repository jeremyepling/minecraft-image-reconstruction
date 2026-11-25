"""Stage 1 evaluation. Camera poses and visible-face labels enter only here."""
from collections import defaultdict
from itertools import groupby, permutations

import numpy as np

from benchmark.camera_labels import screen_matrix
from calibration.world import GROUND
from engine.camera_grid import plane_coordinates
from engine.geometry import project


def truth_camera(pose):
    Q = screen_matrix(pose)
    Q[:, 3] -= Q[:, :3] @ pose['eye_world']
    return Q


def registration(P, pose):
    """Identify compass and integer origin, without optimizing against a building."""
    w, h = pose['image_size']
    Q = truth_camera(pose)
    change = np.linalg.solve(Q[:, :3], P[:, :3])
    same = abs(change[0, 0]) + abs(change[2, 2])
    swapped = abs(change[2, 0]) + abs(change[0, 2])
    T = np.eye(4)
    T[:3, :3] = 0
    T[1, 1] = 1
    for local, world in zip((0, 2), (0, 2) if same >= swapped else (2, 0)):
        T[world, local] = 1 if change[world, local] >= 0 else -1
    T[1, 3] = GROUND + 1
    # Use the same reference ray for every method even if its local origin is far away.
    ref = np.array([[w * .5, h * .72]])
    wxz = plane_coordinates(Q, ref, 1, GROUND + 1)[0]
    lxz = plane_coordinates(P, ref, 1, 0)[0]
    T[[0, 2], 3] = np.rint(wxz - T[np.ix_([0, 2], [0, 2])] @ lxz)
    return T


def observation(faces):
    """Unique visible vertices and uninterrupted collinear unit edges on each plane."""
    vertices = {}
    tracks = defaultdict(set)
    patches = defaultdict(set)
    for face in faces:
        normal = tuple(face['normal'])
        axis = next(i for i, d in enumerate(normal) if d)
        corners = [tuple(map(int, p)) for p in face['world_corners']]
        visible = face['corner_visible']
        for i, a in enumerate(corners):
            if not visible[i]:
                continue
            vertices[a] = face['image_corners'][i]
            patches[axis, a[axis]].add(a)
            j = (i + 1) % 4
            if not visible[j] or face['visible_fraction'] < .98:
                continue
            b = corners[j]
            direction = next(k for k in range(3) if a[k] != b[k])
            anchor = list(a)
            anchor[direction] = 0
            tracks[axis, tuple(anchor), direction].add(min(a[direction], b[direction]))
    spans = []
    for (axis, anchor, direction), units in sorted(tracks.items()):
        for _, group in groupby(enumerate(sorted(units)), lambda pair: pair[1] - pair[0]):
            run = [p for _, p in group]
            if len(run) < 3:
                continue
            a, b = list(anchor), list(anchor)
            a[direction], b[direction] = run[0], run[-1] + 1
            spans.append((axis, direction, tuple(a), tuple(b)))
    return vertices, spans, patches


def project_visible(P, xyz):
    xyz = np.asarray(xyz)
    front = xyz @ P[2, :3] + P[2, 3] > 1.e-6
    xy = np.full((len(xyz), 2), np.nan)
    xy[front] = project(P, xyz[front])
    return xy


def proposal_scores(prediction, T, patches):
    """Image-only extent estimates for the most exposed vertical plane of each axis."""
    rows = []
    for axis in (0, 2):
        family = [(key, pts) for key, pts in patches.items() if key[0] == axis and len(pts) >= 4]
        if not family:
            continue
        (axis, offset), pts = max(family, key=lambda item: len(item[1]))
        free = [i for i in range(3) if i != axis]
        expected = np.ptp(np.asarray(sorted(pts))[:, free], axis=0)
        if not np.all(expected):
            continue
        row = dict(axis=axis, plane=offset, expected=expected.tolist())
        for plane in prediction['planes'] if T is not None else []:
            local = plane['fixed_axis']
            if abs(T[axis, local]) < .5:
                continue
            level = T[axis, 3] + T[axis, local] * plane['offset']
            sizes = np.zeros(3)
            sizes[plane['axes']] = plane['span_blocks']
            inferred = (np.abs(T[:3, :3]) @ sizes)[free]
            row.update(predicted=inferred.tolist(), plane_error_blocks=abs(float(level - offset)),
                       extent_mae_blocks=float(np.abs(inferred - expected).mean()))
        rows.append(row)
    return rows


def score(prediction, pose, faces):
    vertices, spans, patches = observation(faces)
    row = dict(vertices=len(vertices), spans=len(spans), within_2px=0, projected=0, measured_spans=0,
               mean_px=None, p90_px=None, span_mae=None, relative_span_error=None)
    T = None
    row.update(axis_mae_deg=None, fov_error_deg=None)
    axes = prediction.get('axes_homogeneous', [])
    if len(axes) == 3:
        w, h = pose['image_size']
        f = h / (2 * np.tan(np.radians(pose['vertical_fov_deg']) / 2))
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
        predicted = np.linalg.solve(K, np.array(axes).T)
        actual = np.linalg.solve(K, truth_camera(pose)[:, :3])
        predicted /= np.linalg.norm(predicted, axis=0)
        actual /= np.linalg.norm(actual, axis=0)
        angles = np.degrees(np.arccos(np.clip(abs(predicted.T @ actual), 0, 1)))
        row['axis_mae_deg'] = min(float(angles[range(3), perm].mean()) for perm in permutations(range(3)))
    if prediction['P'] is not None:
        P = np.array(prediction['P'], dtype=float)
        optics = P[:, :3] @ P[:, :3].T
        optics /= optics[2, 2]
        fy = np.sqrt(max(0., optics[1, 1] - optics[1, 2] ** 2))
        row['fov_error_deg'] = abs(float(np.degrees(2 * np.arctan2(pose['image_size'][1], 2 * fy)))
                                   - pose['vertical_fov_deg'])
        T = registration(P, pose)
        W = P @ np.linalg.inv(T)
        if vertices:
            xy = project_visible(W, list(vertices))
            error = np.linalg.norm(xy - list(vertices.values()), axis=1)
            error = error[np.isfinite(error)]
            if error.size:
                row.update(projected=len(error), within_2px=int((error <= 2).sum()),
                           mean_px=float(error.mean()), p90_px=float(np.percentile(error, 90)))
        errors, relative = [], []
        for axis, direction, a, b in spans:
            ab = np.array([vertices[a], vertices[b]])
            coords = plane_coordinates(W, ab, axis, a[axis])
            i = [k for k in range(3) if k != axis].index(direction)
            expected = b[direction] - a[direction]
            measured = abs(coords[1, i] - coords[0, i])
            if np.isfinite(measured):
                errors.append(abs(measured - expected))
                relative.append(measured / expected - 1)
        if errors:
            row.update(measured_spans=len(errors), span_mae=float(np.mean(errors)),
                       relative_span_error=float(np.mean(relative)))
    row['charts'] = proposal_scores(prediction, T, patches)
    row['hit_rate'] = row['within_2px'] / max(1, row['vertices'])
    return row


def summarize(rows):
    layouts = sorted({r['layout'] for r in rows})
    def layout_mean(key):
        means = []
        for layout in layouts:
            values = [r[key] for r in rows if r['layout'] == layout and r[key] is not None]
            if values:
                means.append(np.mean(values))
        return float(np.mean(means)) if means else None
    charts = [c for r in rows for c in r['charts']]
    extents = [c['extent_mae_blocks'] for c in charts if 'predicted' in c]
    return dict(layouts=len(layouts), images=len(rows), metric_images=sum(r['metric'] for r in rows),
                vertices=sum(r['vertices'] for r in rows), within_2px=sum(r['within_2px'] for r in rows),
                projected=sum(r['projected'] for r in rows), spans=sum(r['spans'] for r in rows),
                measured_spans=sum(r['measured_spans'] for r in rows), layout_hit_rate=layout_mean('hit_rate'),
                layout_mean_px=layout_mean('mean_px'), layout_span_mae_blocks=layout_mean('span_mae'),
                layout_axis_mae_deg=layout_mean('axis_mae_deg'), layout_fov_error_deg=layout_mean('fov_error_deg'),
                charts=len(charts), predicted_charts=len(extents),
                conditional_chart_extent_mae=float(np.mean(extents)) if extents else None,
                exact_chart_planes=sum(c.get('plane_error_blocks') == 0 for c in charts))
