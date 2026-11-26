"""Single-image Stage 1. All coordinates use source-frame pixels and full blocks."""
import json
import numpy as np
from PIL import Image, ImageDraw

from benchmark.oblique_recess import pavement
from engine.plane_candidates import foreground
from engine.geometry import project


METHODS = ('pavement', 'seam-fit', 'tile-lines', 'seam-or-atlas')
DEFAULT = 'seam-or-atlas'
PAVEMENT_SETTINGS = dict(implementation='benchmark.oblique_recess.pavement@a67988b',
                         profile_rows=26, floor_band=[.81, .985], minimum_height_ratio=.90)


def pavement_camera(image):
    """Expose the camera of the published voxel fitter without fitting occupancy."""
    fit = pavement(np.asarray(image.convert('RGB')))
    if fit is None:
        return dict(P=None, scale_status='uncertain', reason='Two consistent pavement periods were unavailable.')
    K, _, directions, up, ground, height, offsets, info = fit
    basis = np.column_stack((directions[0], up, directions[1]))
    # Shift by integer blocks to keep the origin in the shot. This changes only
    # the choice of local coordinate labels in the published camera.
    ref = ground(np.array([[image.width / 2, .72 * image.height]])) * height
    floors = (ref @ basis)[0, [0, 2]]
    origin_x, origin_z = np.rint(floors - offsets) + offsets
    origin = basis @ [origin_x, -height, origin_z]
    P = K @ np.column_stack((basis, origin))
    return dict(P=P.tolist(), scale_status='supported', evidence=info,
                reason='Two pavement periods passed the published camera fit; each repeat is taken as one block.')


def plane_coordinates(P, xy, fixed_axis, offset):
    """Intersect source pixels with one lattice plane, preserving fractional units."""
    free = [axis for axis in range(3) if axis != fixed_axis]
    H = np.column_stack((P[:, free], P[:, 3] + offset * P[:, fixed_axis]))
    rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(H).T
    return rays[:, :2] / rays[:, 2, None]


def contact_planes(image, P):
    """Locate vertical support planes using repeated ground contacts of the silhouette.

    A proposal describes a rectangular coordinate chart. Cells, occupied areas
    and depth behind the chart are left to the face stage.
    """
    _, mask = foreground(image)
    yy, xx = np.where(mask)
    if not len(xx) or mask.mean() > .95:
        return []
    cols = np.unique(xx)
    scale = np.array(image.size) / mask.shape[::-1]
    foot = np.column_stack((cols + .5, mask.shape[0] - mask[::-1, cols].argmax(0))) * scale
    head = np.column_stack((cols + .5, mask[:, cols].argmax(0))) * scale
    xz = plane_coordinates(P, foot, 1, 0)
    planes = []
    for dim, axis in enumerate((0, 2)):
        vals = xz[:, dim]
        eligible = np.isfinite(vals) & (np.abs(vals) < 256)
        rounded = np.rint(vals[eligible]).astype(int)
        depths, counts = np.unique(rounded, return_counts=True)
        if not len(depths):
            continue
        depth = int(depths[np.argmax(counts)])
        keep = np.abs(vals - depth) < .45
        if keep.sum() < max(4, .08 * len(cols)):
            continue
        uv = plane_coordinates(P, np.vstack((foot[keep], head[keep])), axis, depth)
        if not np.isfinite(uv).all():
            continue
        lo, hi = np.rint(np.quantile(uv, [.015, .985], axis=0)).astype(int)
        if np.any(hi - lo < 1) or np.any(hi - lo > 128):
            continue
        free = [i for i in range(3) if i != axis]
        xyz = np.full((4, 3), depth)
        xyz[:, free] = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]
        planes.append(dict(fixed_axis=axis, offset=depth, axes=free, low=lo.tolist(), high=hi.tolist(),
                           span_blocks=(hi-lo).tolist(), quad=project(P, xyz).tolist(),
                           ground_contact_samples=int(keep.sum())))
    return planes


def predict(image, version='26.3', method=None, *, atlas_path=None):
    if version != '26.3':
        raise ValueError('The Stage 1 camera estimators are calibrated for Java 26.3.')
    method = method or DEFAULT
    if method == 'pavement':
        result = pavement_camera(image)
        result['settings'] = PAVEMENT_SETTINGS
    elif method == 'seam-or-atlas':
        from engine.camera_metric import predict as fit
        from engine.camera_seam import SETTINGS as seam_settings
        result = fit(image, version, atlas_path=atlas_path)
        result['settings'] = dict(primary='seam-fit', fallback='atlas-floor', seam_fit=seam_settings,
                                  atlas_version=version)
    elif method == 'seam-fit':
        from engine.camera_seam import predict as fit
        result = fit(image, version)
    elif method == 'tile-lines':
        from engine.camera_lines import predict as fit
        result = fit(image, version)
    else:
        raise ValueError(f'Unknown camera method {method}')
    result.update(method=method, version=version, image_size=list(image.size))
    if result['P'] is None:
        from engine.cube_axes import fit_axes
        from engine.structural_lines import detect
        result['axes_homogeneous'] = result.get('axes_homogeneous') or fit_axes(detect(image), image.size)
        result['planes'] = []
    else:
        P = np.asarray(result['P'], dtype=float)
        result['axes_homogeneous'] = P[:, :3].T.tolist()
        result['planes'] = contact_planes(image, P)
    return result


def build_camera(image_path, version, out, *, atlas_path=None):
    from engine.atlas import digest
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    camera = predict(image, version, atlas_path=atlas_path)
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(version=version, image_sha256=digest(image_path),
                      through='camera', stages=dict(camera=camera['method']),
                      scale_status=camera['scale_status'])
    (out / 'camera.json').write_text(json.dumps(camera, indent=2, allow_nan=False))
    (out / 'build-manifest.json').write_text(json.dumps(provenance, indent=2))
    pen = ImageDraw.Draw(image)
    P = np.array(camera['P'])
    for chart in camera['planes']:
        free = chart['axes']
        for axis in (0, 1):
            other = 1 - axis
            for index in range(chart['low'][axis], chart['high'][axis] + 1):
                points = np.full((2, 3), chart['offset'])
                points[:, free[axis]] = index
                points[:, free[other]] = [chart['low'][other], chart['high'][other]]
                if np.all(points @ P[2, :3] + P[2, 3] > 0):
                    pen.line([tuple(p) for p in project(P, points)], fill=(46, 219, 190), width=2)
    image.save(out / 'grid.png')
    print(f"{version} camera: {camera['method']} ({camera['scale_status']}) -> {out / 'camera.json'}")
