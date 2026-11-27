"""Score native visible corners after fitting one integer origin on disjoint corners."""

from itertools import product
import numpy as np
from PIL import Image, ImageDraw

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


def draw_registration_example(row, result, output):
    folder = SOURCE / row["dataset"] / "data" / row["split"] / "frames" / row["id"]
    pose, faces = read(folder / "camera.json"), read(folder / "labels.json")["faces"]
    points = {tuple(p): uv for face in faces for p, uv, seen in
              zip(face["corners_world"], face["corners_px"], face["corner_visible"]) if seen}
    keys = np.array(sorted(points)[1::2])
    # Use the same spatial sample of withheld corners in both panels.
    selected = keys[np.linspace(0, len(keys) - 1, 26).astype(int)]
    expected = [points[tuple(p)] for p in selected]
    camera = np.array(row["projection"]["P"])
    matrix = camera @ np.linalg.inv(registration(camera, pose))
    image = Image.open(folder / "rgb.png").convert("RGB")
    sheet = Image.new("RGB", (1440, 524), "#152224")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (shift, title) in enumerate([
            ([0, 0, 0], "Ground-ray registration"),
            (result["integer_world_xyz_offset_from_registered"], "One shared integer XYZ translation")]):
        tile = image.copy()
        draw = ImageDraw.Draw(tile)
        for uv, projected in zip(expected, project_points(matrix, selected + shift)):
            x, y = uv
            u, v = projected
            draw.line([(x, y), (u, v)], fill="#fff4c2", width=3)
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="#30ffd0")
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill="#ff703c")
        sheet.paste(tile.resize((720, 450), Image.Resampling.LANCZOS), (720 * index, 36))
        sheet_draw.text((12 + 720 * index, 6), title, font_size=21, fill="white")
    sheet_draw.text((12, 491), "Castle 03 limestone rear view. Turquoise is the recorded visible corner. Orange is the MoGe projection.",
                    font_size=18, fill="white")
    sheet.save(output, quality=90)


def main():
    reviews = ROOT / "benchmark/reviews"
    moge = read(reviews / "camera-moge-surface-first-01/proposals.json.gz")
    joint = read(reviews / "camera-grid-joint-01/cases.json.gz")
    cases = []
    for row in moge:
        if row["projection"]:
            result = grade(row, row["projection"])
            cases.append(dict(id=row["id"], method="MoGe", **result))
            if row["id"] == "test-castle-03-limestone-rear-oblique":
                draw_registration_example(row, result, reviews / "camera-recovery-next-01/castle-registration.jpg")
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
