"""Full camera-handoff replay and separate native visible-corner evaluation."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark.camera_metrics import registration, truth_camera
from benchmark.expanded_baseline import read, replay_source_fields
from benchmark.expanded_scores import score_build
from calibration.world import GROUND
from engine.geometry import project


def camera_validation(entry, camera):
    folder = Path(entry['frame'])
    pose = read(folder / 'camera.json')
    faces = read(folder / 'labels.json')['faces']
    P = np.asarray(camera['P'])
    transform = registration(P, pose)
    aligned = P @ np.linalg.inv(transform)
    true = truth_camera(pose)
    w, h = pose['image_size']
    focal = h / (2 * np.tan(np.deg2rad(pose['vertical_fov_deg']) / 2))
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
    Q = P[:, :3] @ P[:, :3].T
    Q /= Q[2, 2]
    cx, cy = Q[:2, 2]
    fx, fy = np.sqrt(np.maximum(0., [Q[0, 0]-cx*cx, Q[1, 1]-cy*cy]))
    estimated_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    a, b = np.linalg.solve(estimated_K, aligned[:, :3]), np.linalg.solve(K, true[:, :3])
    axis = np.sum(a*b, axis=0) / (np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0))
    axes = np.rad2deg(np.arccos(np.clip(abs(axis), 0, 1)))
    inferred_height = -np.linalg.solve(P[:, :3], P[:, 3])[1]
    height = pose['eye_world'][1] - (GROUND + 1)
    vertices = {tuple(c): xy for face in faces for c, xy, visible in
                zip(face['corners_world'], face['corners_px'], face['corner_visible']) if visible}
    projected = project(aligned, list(vertices)) if vertices else np.empty((0, 2))
    errors = np.linalg.norm(projected - np.array(list(vertices.values())).reshape(-1, 2), axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite visible corner reprojection")
    return dict(fov_error_deg=abs(float(np.rad2deg(2*np.arctan2(h, 2*fy))) - pose['vertical_fov_deg']),
                axis_errors_deg=axes.tolist(), ground_height_error_blocks=float(inferred_height-height),
                relative_scale_error=float(inferred_height/height-1), vertices=len(errors),
                median_reprojection_px=float(np.median(errors)) if len(errors) else None,
                p90_reprojection_px=float(np.percentile(errors, 90)) if len(errors) else None,
                within_2px=int((errors<=2).sum()))


def replay(entry, camera, baseline_folder, output_folder):
    from engine import surface_model, visible_placement
    image = Path(entry['frame']) / 'rgb.png'
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with replay_source_fields(image, Path(baseline_folder)), patch.object(visible_placement, 'seam_camera', return_value=camera):
        surface_model.build(image, '26.3', output_folder, through='placement')
    manifest = read(output_folder / 'build-manifest.json')
    if manifest['placement']['P'] != camera['P']:
        raise ValueError('Supplied camera did not reach exported placement')
    case = score_build(entry, output_folder)
    if camera['P'] is not None:
        case['camera_validation'] = camera_validation(entry, camera)
    return case
