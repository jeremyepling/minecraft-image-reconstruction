"""Measure vertical first hits and two controlled same-pavement negatives on one photo."""
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.wall_floor_counterfactual import recover, table
from benchmark.wall_source_projection import Scene, STEP, decoded_pack, framed, read, save


def axes(scene, cell):
    i = scene.cells.index(cell)
    return np.bincount(scene.axes[scene.owned[i]], minlength=3).tolist()


def pavement_controls(image, parts, camera, prior, cell):
    scene = Scene(image, parts, camera, prior)
    # Obtain the source pavement pattern on distant background, excluding all
    # predicted surfaces and occupied pixels. Keep the camera and prior fixed.
    free = np.flatnonzero(~scene.union & (scene.owner < 0))
    rgb = scene.rgb.reshape(-1, 3)[free]
    y, x = np.divmod(free, scene.shape[1])
    rays = np.column_stack((STEP*x+.5, STEP*y+.5, np.ones(len(x)))) @ scene.inv.T
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = -scene.eye[1] / rays[:, 1]
    close = np.isfinite(depth) & (depth > 2) & (depth < 50)
    free = free[close]
    phase = scene.uv(free, (0, -1, 0), 1)
    tile, known = table(phase, rgb[close], np.ones(len(free), bool))
    assert known.all()

    # Replace the real photograph at every first-hit sample of a disputed
    # 3D block with an actual cube made of the same pavement material.
    # Rendering uses each visible face's true plane. Flat lighting maximizes
    # the chance that foreground and pavement could be confused.
    n = scene.cells.index(cell)
    idx = scene.owned[n]
    altered = np.array(image)
    for a in np.unique(scene.axes[idx]):
        pix = idx[scene.axes[idx] == a]
        yy, xx = np.divmod(pix, scene.shape[1])
        altered[STEP*yy, STEP*xx] = tile[scene.uv(pix, cell, a)]
    kept, _ = recover(Image.fromarray(altered), parts, None, camera, prior)
    wall = dict(cell=cell, axis_samples=axes(scene, cell), retained=cell in kept)

    # Choose a visible patch whose 3x3 top face footprint lies wholly outside
    # both the build masks and projection. Elevation -1 puts the pad top on the
    # actual inferred ground plane y=0. Original source pavement therefore is
    # exactly the intended visible appearance, including lighting and filtering.
    floor = None
    for x in range(-12, 13):
        for z in range(-12, 13):
            pad = {(x+i, -1, z+j) for i in (-1,0,1) for j in (-1,0,1)}
            tops = [idx[a==1] for c in pad for idx,a,_ in [scene.faces(c)]]
            footprint = np.unique(np.concatenate(tops))
            if (min(map(len, tops)) < 32 or len(footprint) < 450
                    or np.any(scene.union[footprint] | (scene.owner[footprint] >= 0))):
                continue
            test_scene = Scene(image, parts, camera, prior | pad)
            core=(x, -1, z)
            if axes(test_scene, core)[1] < 32:
                continue
            # Let upstream surface finding explicitly identify the pad. A
            # downstream floor test should preserve its visible top squares.
            top_mask = np.zeros(scene.shape, bool)
            top_mask.ravel()[footprint] = True
            runs = []
            for yy, scan in enumerate(top_mask):
                edges = np.flatnonzero(np.diff(np.pad(scan.astype(int), 1)))
                runs.extend((yy, int(a), int(b)) for a, b in edges.reshape(-1, 2))
            pad_part = dict(support=dict(size=list(scene.shape[::-1]), runs=runs))
            kept, _ = recover(image, parts+[pad_part], None, camera, prior | pad)
            removed = sorted(pad-kept)
            floor = dict(center=core, removed=[dict(cell=c,axis_samples=axes(test_scene,c)) for c in removed],
                         center_retained=core in kept, center_axis_samples=axes(test_scene, core),
                         vertical_gate_wrong_deletions=sum(max(axes(test_scene,c)[::2])>=16 for c in removed))
            break
        if floor is not None:
            break
    assert floor is not None
    return dict(same_pavement_wall=wall, true_flush_pad=floor)


def main():
    root = Path('runs/floor')
    control = Path('../wall-section-round/runs/control')
    changed = {(r['split'], r['id']) for r in read(root/'cases.json') if r['changed']}
    rows = []
    controls = None
    for split, f, source in framed.cases(['native', 'fresh', 'extra']):
        if (split, f['id']) not in changed:
            continue
        stem = Path(split)/f['id']/'reconstruction.zip'
        prior = set(decoded_pack(control/stem))
        kept = set(decoded_pack(root/stem))
        image, parts, _, camera = framed.photo(split, f, source)
        scene = Scene(image, parts, camera, prior)
        rows.append(dict(id=f['id'], cells=[dict(cell=c, axis_samples=axes(scene,c)) for c in sorted(prior-kept)]))
        if split=='fresh':
            cell=max(prior-kept,key=lambda c:max(axes(scene,c)[::2]))
            controls=pavement_controls(image,parts,camera,prior,cell)
    assert controls is not None
    save(Path('benchmark/reviews/wall-source-projection-01/floor-controls.json'),
         dict(measured=rows, counterfactuals=controls))


if __name__=='__main__':
    main()
