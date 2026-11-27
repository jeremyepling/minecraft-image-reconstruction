"""Build native mask options using only the source screenshot and its frozen model fields."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from engine.camera_lines import _runs
from engine.floating_sky import sky_support
from engine.mask_first import _image_depth
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


OUTPUT = ROOT / 'runs/sky-crop-chooser'
SKY = SOURCE / 'floating-build-grid-01'
ACTIVE = WORKTREES / 'floating-visible-builds/runs/floating-cloud-gate-01/sky'
OPTIONS = ('sky-trim', 'colour-edge', 'depth-normal', 'dominant-piece', 'planarity')


def native(support, size):
    out = np.zeros(size[::-1], bool)
    crop = _native_alpha(support, size)
    if crop is not None:
        alpha, (l, t, r, b) = crop
        out[t:b, l:r] = np.asarray(alpha) != 0
    return out


def fields(folder, size):
    receipt = read(folder / 'receipt.json')
    values = {}
    for key in ('normal', 'depth', 'planarity'):
        name = f'{key}.npz'
        path = folder / 'handoffs' / name
        if digest(path) != receipt['handoffs_sha256'][name]:
            raise ValueError(f'Changed source model field {path}')
        with np.load(path, allow_pickle=False) as archive:
            values[key] = archive['relaxed' if key == 'planarity' else 'value']
    normal = np.stack([np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.BILINEAR))
                       for channel in values['normal'].transpose(2, 0, 1)], axis=-1)
    return normal, _image_depth(values['depth'], size), np.asarray(
        Image.fromarray(values['planarity']).resize(size, Image.Resampling.NEAREST))


def colour_edges(seed, rgb, sky):
    """Reclassify a two-pixel collar from nearby inside/outside colours."""
    mask = Image.fromarray(seed)
    inside = np.asarray(mask.filter(ImageFilter.MinFilter(5)))
    outside = ~np.asarray(mask.filter(ImageFilter.MaxFilter(5)))
    y, x = np.nonzero(~(inside | outside | sky))
    best_in, best_out = np.full(len(x), 1000.), np.full(len(x), 1000.)
    for radius in (2, 4, 6):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
            yy = np.clip(y + dy * radius, 0, seed.shape[0] - 1)
            xx = np.clip(x + dx * radius, 0, seed.shape[1] - 1)
            dist = np.max(np.abs(rgb[yy, xx] - rgb[y, x]), axis=1) + radius / 2.
            best_in = np.minimum(best_in, np.where(inside[yy, xx], dist, 1000.))
            best_out = np.minimum(best_out, np.where(outside[yy, xx], dist, 1000.))
    result = seed.copy()
    change = np.abs(best_in - best_out) > 3
    result[y[change], x[change]] = best_in[change] < best_out[change]
    return result & ~sky


def depth_normal(seed, normal, depth, sky):
    """Refit an affine inverse-depth sheet and check direction in a six-pixel collar."""
    y, x = np.nonzero(seed)
    if len(x) < 20:
        return seed
    idx = np.linspace(0, len(x) - 1, min(1500, len(x)), dtype=int)
    y, x = y[idx], x[idx]
    design = np.column_stack((x, y, np.ones_like(x)))
    z = depth[y, x]
    use = np.ones(len(x), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(design[use], z[use], rcond=None)[0]
        errors = np.abs(z - design @ coef)
        use = errors <= max(.007, np.quantile(errors, .7))
    direction = np.median(normal[y[use], x[use]], axis=0)
    direction /= max(np.linalg.norm(direction), 1.e-6)
    sy, sx = np.nonzero(np.asarray(Image.fromarray(seed).filter(ImageFilter.MaxFilter(13))) & ~sky)
    residual = abs(depth[sy, sx] - (coef[0]*sx + coef[1]*sy + coef[2]))
    keep = (residual <= max(.014, np.median(errors)*3)) & ((normal[sy, sx] @ direction) >= .90)
    result = np.zeros_like(seed)
    result[sy[keep], sx[keep]] = True
    return result


def largest(mask):
    components = _runs(mask)
    result = np.zeros_like(mask)
    for y, x0, x1 in max(components, key=lambda runs: sum(b-a for _,a,b in runs), default=[]):
        result[y, x0:x1] = True
    return result


def one_frame(folder):
    case = folder.name
    active = ACTIVE / case / 'source-handoff'
    path = SKY / 'images' / case / 'sky.png'
    sha = digest(path)
    if any(read(p)[k] != sha for p, k in [(active/'build-manifest.json','image_sha256'),
                                           (folder/'receipt.json','source_sha256')]):
        raise ValueError(f'Image hash mismatch for {case}')
    image = Image.open(path).convert('RGB')
    rgb = np.asarray(image, np.int16)
    normals, depth, labels = fields(folder, image.size)
    # The row-colour test also detects clear sky seen through enclosed openings.
    row_agrees = (rgb[:, 0] == rgb[:, -1]).all(axis=1) & (rgb[:, 0, 2] - rgb[:, 0, 0] > 40)
    sky = sky_support(image) | (row_agrees[:, None] & (rgb == rgb[:, :1]).all(axis=2))
    areas = np.bincount(labels.ravel())
    predictions = {name: {} for name in OPTIONS}
    for part in read(active/'surfaces.json')['planes']:
        original = native(part['support'], image.size)
        trimmed = original & ~sky
        hits = np.bincount(labels[trimmed], minlength=len(areas))
        iou = hits / np.maximum(1, areas + trimmed.sum() - hits)
        iou[0] = 0
        candidate = (labels == iou.argmax()) & ~sky if iou.max() >= .08 else trimmed
        # Expensive local filters operate in a padded source ROI. Planarity remains full-frame.
        y, x = np.nonzero(original)
        roi = np.s_[max(0, y.min()-8):y.max()+9, max(0, x.min()-8):x.max()+9]
        options = {'sky-trim': trimmed, 'planarity': candidate}
        for name, local in [('colour-edge', colour_edges(trimmed[roi], rgb[roi], sky[roi])),
                            ('depth-normal', depth_normal(trimmed[roi], normals[roi], depth[roi], sky[roi])),
                            ('dominant-piece', largest(trimmed[roi]))]:
            mask = np.zeros_like(original)
            mask[roi] = local
            options[name] = mask
        for name, mask in options.items():
            if not np.array_equal(original, mask):
                predictions[name][part['id']] = dict(size=list(image.size), runs=row_runs(mask))
    proof = dict(image_sha256=sha, active_surfaces_sha256=digest(active/'surfaces.json'),
                 fields_sha256={k: v for k,v in read(folder/'receipt.json')['handoffs_sha256'].items()
                                if k in ('depth.npz','normal.npz','planarity.npz')})
    return case, predictions, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    options = {name: {} for name in OPTIONS}
    manifest = {}
    for folder in sorted((SKY/'baseline/sky').glob('*/*')):
        case, result, manifest[folder.name] = one_frame(folder)
        for name in OPTIONS:
            options[name][case] = result[name]
        print(case, {n:len(result[n]) for n in OPTIONS}, flush=True)
    for name in OPTIONS:
        save(args.output / f'{name}.json', dict(frames=options[name]))
    save(args.output/'sources.json', manifest)


if __name__ == '__main__':
    main()
