"""Score native visible corners after fitting one integer origin on disjoint corners."""

from itertools import product
import numpy as np

from benchmark.camera_metrics import registration, truth_camera
from benchmark.expanded_baseline import ROOT, SOURCE, read, save


def project_points(matrix, points):
    homogeneous = points @ matrix[:, :3].T + matrix[:, 3]
    if (homogeneous[:, 2] <= 0).any():
        raise ValueError("Visible points fell behind the proposed camera")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def native_corner_score(matrix, pose, faces):
    """Use only the orientation convention from registration and allow one XYZ translation."""
    matrix = np.asarray(matrix)
    convention = registration(matrix, pose)
    aligned = matrix @ np.linalg.inv(convention)
    vertices = {tuple(xyz): uv for face in faces for xyz, uv, seen in
                zip(face["corners_world"], face["corners_px"], face["corner_visible"]) if seen}
    # Score displacement in local projected block lengths, allowing the longest
    # mathematical unit vector at each already-visible vertex (conservative).
    points = np.asarray(sorted(vertices))
    observed = np.asarray([vertices[tuple(point)] for point in points])
    # Interleave sorted distinct physical vertices for spatially distributed fit and test sets.
    fit = points[::2]
    targets = observed[::2]
    homogeneous = fit @ aligned[:, :3].T + aligned[:, 3]
    rows = aligned[:2, :3] - targets[:, :, None] * aligned[2, :3]
    rhs = targets * homogeneous[:, 2, None] - homogeneous[:, :2]
    start = np.linalg.lstsq(rows.reshape(-1, 3), rhs.ravel(), rcond=None)[0]
    if np.median(homogeneous[:, 2] + aligned[2, :3] @ start) < 0:
        aligned = -aligned

    def residual(delta):
        return (project_points(aligned, fit + delta) - targets).ravel()

    refined = start
    for _ in range(12):
        projected = project_points(aligned, fit + refined)
        depth = (fit + refined) @ aligned[2, :3] + aligned[2, 3]
        jacobian = (aligned[:2, :3] - projected[:, :, None] * aligned[2, :3]) / depth[:, None, None]
        step = np.linalg.lstsq(jacobian.reshape(-1, 3), (targets - projected).ravel(), rcond=None)[0]
        if np.linalg.norm(step) < 1.e-6:
            break
        refined = refined + step
    center = np.rint(refined).astype(int)
    # Find the integer translation with least pixel RMS near the continuous solution.
    candidates = [center + step for step in product(range(-2, 3), repeat=3)]
    delta = min(candidates, key=lambda d: np.mean(residual(d) ** 2))
    held = points[1::2]
    distances = np.linalg.norm(project_points(aligned, held + delta) - observed[1::2], axis=1)
    true = truth_camera(pose)
    pixel_width = np.max([np.linalg.norm(project_points(true, held + ax) - observed[1::2], axis=1)
                          for ax in np.eye(3)], axis=0)
    return {
        "fitted_vertices": len(fit), "check_vertices": len(distances),
        "integer_world_xyz_offset_from_registered": delta.tolist(),
        "median_check_px": float(np.median(distances)), "p90_check_px": float(np.percentile(distances, 90)),
        "median_check_in_longest_projected_unit": float(np.median(distances / pixel_width)),
        "p90_check_in_longest_projected_unit": float(np.percentile(distances / pixel_width, 90)),
        "check_within_2px": int((distances <= 2).sum()),
    }


def grade(row, camera):
    frame = SOURCE / row["dataset"] / "data" / row["split"] / "frames" / row["id"]
    return native_corner_score(camera["P"], read(frame / "camera.json"), read(frame / "labels.json")["faces"])


def main():
    reviews = ROOT / "benchmark/reviews"
    moge = read(reviews / "camera-moge-surface-first-01/proposals.json.gz")
    joint = read(reviews / "camera-grid-joint-01/cases.json.gz")
    cases = []
    for row in moge:
        if row["projection"]:
            cases.append(dict(id=row["id"], method="MoGe", **grade(row, row["projection"])))
    for row in joint:
        for index, option in enumerate(row.get("prediction", {}).get("best", []), 1):
            cases.append(dict(id=row["id"], split=row["split"], method=f"joint{index}",
                              exported_for_diagnostic=bool(row.get("ablations")), **grade(row, option["camera"])))

    # Reference-only invariance controls include a large shift in every axis.
    sample = next(r for r in joint if r.get("ablations"))
    folder = SOURCE / sample["dataset"] / "data" / sample["split"] / "frames" / sample["id"]
    pose, faces = read(folder / "camera.json"), read(folder / "labels.json")["faces"]
    control = []
    for offset in [[0, 0, 0], [123, -27, 48]]:
        shifted = truth_camera(pose)
        shifted[:, 3] += shifted[:, :3] @ np.asarray(offset)
        result = native_corner_score(shifted, pose, faces)
        assert result["p90_check_px"] < 1.e-7
        control.append(dict(input_offset=offset, **result))
    save(reviews / "camera-recovery-next-01/relative-calibration.json", dict(translation_controls=control, cases=cases))


if __name__ == "__main__":
    main()
