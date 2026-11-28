"""Test whether source normals and local inverse-depth planes support a measured wall grid."""
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import NATIVE, cases
from engine.geometry import bilinear, project
from engine.surface_fields import MODELS

OUT = ROOT / 'runs/patch-planes'
FRESH = WORKTREES / 'floating-fresh-validation/runs/floating-fresh'
NEW = WORKTREES / 'floating-patch-validation/runs/floating-patch-validation'
SKY = WORKTREES / 'floating-horizontal-bridges/runs/six-sky'
FIXED = WORKTREES / 'floating-visible-builds/runs/floating-cloud-gate-01'
COHORTS = ('original', 'fresh', 'supplemental', 'sky', 'new', 'new2')
CANDIDATES = ('planes', 'opposed')
OFFSETS = np.stack(np.meshgrid([.22, .5, .78], [.22, .5, .78]), -1).reshape(-1, 2)


def inputs(cohort):
    if cohort == 'original':
        ordinary = NATIVE.parent / 'source'
        whole = WORKTREES / 'expanded-wall-rulers/runs/wall-rulers/source_material'
        return ordinary, ordinary/'atlas/connected', whole/'charts', whole/'assembly', list(cases())
    if cohort == 'sky':
        return FIXED, FIXED/'garden-anchor-extension/connected', SKY/'charts', SKY/'whole-wall', read(assembly.SHARED/'fixtures.json')['cases']
    root = NEW if cohort in ('new', 'new2') else FRESH
    manifest = 'supplemental-manifest.json' if cohort == 'supplemental' else 'manifest.json'
    if cohort == 'new2':
        manifest = 'holdout2-manifest.json'
    return root/'ordinary', root/'ordinary/atlas/connected', root/'source_material/charts', root/'source_material/assembly', read(root/manifest)['cases']


def field_files(cohort, case):
    group = Path(case['dataset'])/case['id']
    if cohort == 'sky':
        return assembly.SHARED/'baseline/sky'/group/'handoffs'
    if cohort == 'original':
        archived = WORKTREES/'full-expanded-controls/runs/expanded-baseline-01'/group/'handoffs'
        return archived if archived.is_dir() else NATIVE.parent/'fields'/group
    return (NEW if cohort in ('new', 'new2') else FRESH)/'fields'/group


def field_input(cohort, case, source_sha, handoff):
    folder = field_files(cohort, case)
    hashes = {kind: digest(folder/(kind+'.npz')) for kind in MODELS}
    handoff = read(handoff/'build-manifest.json')
    assert handoff['image_sha256'] == source_sha
    for kind, (_, _, model_sha) in MODELS.items():
        assert handoff['model_sha256'][kind] == model_sha
    receipt = folder/'receipt.json' if (folder/'receipt.json').exists() else folder.parent/'receipt.json'
    if receipt.exists():
        rec = read(receipt)
        assert rec['source_sha256'] == source_sha
        for kind in MODELS:
            assert rec['model_sha256'][kind] == MODELS[kind][2]
            if 'fields_sha256' in rec:
                assert rec['fields_sha256'][kind] == hashes[kind]
    values = {}
    for kind in MODELS:
        with np.load(folder/(kind+'.npz'), allow_pickle=False) as f:
            values[kind] = f['value']
    return values, hashes


def robust_plane(xy, depth):
    """Fit relative inverse depth as an affine function of image coordinates, rejecting step edges."""
    X = np.column_stack((xy, np.ones(len(xy))))
    keep = np.ones(len(xy), bool)
    for _ in range(4):
        beta, _, rank, _ = np.linalg.lstsq(X[keep], depth[keep], rcond=None)
        if rank < 3:
            return None
        residual = depth-X@beta
        mad = np.median(np.abs(residual[keep]-np.median(residual[keep])))
        updated = np.abs(residual) <= max(3.5*1.4826*mad, 1e-6)
        if np.array_equal(updated, keep) or updated.sum() < 8:
            break
        keep = updated
    return beta, float(1.4826*mad)


def sample_cells(H, uv, fields, size):
    pixels = project(H, uv[:,None,:]+OFFSETS)
    xy = pixels[:,4,:]
    normal = fields['normal']
    depth = fields['depth']
    normals = np.median(bilinear(normal, pixels*normal.shape[1::-1]/size), axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    depths = np.median(bilinear(depth[:,:,None], pixels*depth.shape[1::-1]/size)[...,0], axis=1)
    return xy, normals, depths


def seeded_piece(supported, seed, lookup, uv):
    """Reach only wall cells in this patch, across touching visible samples."""
    found = np.zeros(len(supported), bool)
    found[seed] = True
    queue = deque(seed)
    while queue:
        i = queue.popleft()
        u,v = uv[i]
        for key in ((u-1,v),(u+1,v),(u,v-1),(u,v+1)):
            j = lookup.get(key)
            if j is not None and supported[j] and not found[j]:
                found[j] = True
                queue.append(j)
    return found


def patch_wall(chart, original, fields, size):
    H = np.array(chart['H'])
    assert np.array_equal(H, original['H'])
    uv = np.array([(c['u'],c['v']) for c in chart['cells']])
    lookup = {tuple(p):i for i,p in enumerate(uv)}
    seeds = sorted({lookup[(c['u'],c['v'])] for c in original['cells'] if (c['u'],c['v']) in lookup})
    ev = dict(id=chart['id'], whole=len(uv), measured=len(seeds), patches=[])
    if not seeds:
        return chart, ev
    xy, normals, depths = sample_cells(H, uv, fields, size)
    # The chart's inverse homogeneous denominator is proportional to physical
    # inverse camera depth. Choose its sign so inverse depth is positive at the seed.
    inverse = np.linalg.inv(H)[2]
    keep = np.zeros(len(uv), bool)
    ground = set(seeds)
    while ground:
        anchor = min(ground)
        delta = np.max(np.abs(uv - uv[anchor]), axis=1)
        cluster = sorted(i for i in ground if i == anchor or (delta[i]<=3 and normals[i]@normals[anchor] >= .94))
        ground.difference_update(cluster)
        center = np.median(xy[cluster], axis=0)
        rel = xy-center
        close = (delta<=4) & (normals@normals[anchor]>=.94)
        model = robust_plane(rel[close], depths[close]) if close.sum() >= 8 else None
        if model is None:
            # Insufficient sampled area cannot establish contradictory evidence.
            keep[:] = True
            ev['patches'].append(dict(seeds=len(cluster),fit=False))
            break
        beta, noise = model
        z = max(abs(beta[2]), 1e-6)
        slope = inverse[:2]*np.sign(inverse[:2]@center+inverse[2])
        cosine = float(slope@beta[:2]/max(1e-12,np.linalg.norm(slope)*np.linalg.norm(beta[:2])))
        local_change = float(np.ptp(rel[close]@beta[:2]))
        opposed = cosine < -.55 and local_change > max(5*noise, .017*z)
        normal_ok = normals@normals[anchor] >= np.cos(np.radians(30))
        tolerance = max(4.5*noise, .055*z)
        expected = beta[2] + rel@beta[:2]
        plausible = normal_ok & (np.abs(depths-expected) <= max(tolerance, .09*z))
        if opposed:
            # A nonnegative disparity scale cannot follow this slope. Keep the
            # locally indistinguishable depth band, even on a flat-looking facade.
            plausible &= np.abs(depths-beta[2]) <= tolerance
        piece = seeded_piece(plausible, cluster, lookup, uv)
        keep |= piece
        ev['patches'].append(dict(seeds=len(cluster),fit=True,cosine=cosine,opposed=bool(opposed),
                                  relative_noise=noise/z,relative_span=local_change/z,retained=int(piece.sum())))
    assert keep[seeds].all()
    ev['retained'] = int(keep.sum())
    return dict(chart,cells=[c for c,k in zip(chart['cells'],keep) if k]), ev


def predict(cohort, case, candidate):
    source, ordinary, whole, _, _ = inputs(cohort)
    name = case['id']
    rel = Path('sky')/name/'charts.json'
    image = Path(case['variants']['sky']['image'])
    sha = digest(image)
    assert sha == case['variants']['sky']['source_sha256']
    handoff = source/'sky'/name/'source-handoff'
    raw, wall = read(ordinary/rel), read(whole/rel)
    for packet in (raw,wall):
        paths.check_handoff(packet,handoff,sha)
    fields, hashes = field_input(cohort,case,sha,handoff)
    with Image.open(image) as im:
        size=np.array(im.size)
    seeds = {c['id']:c for c in raw['charts']}
    out, evidence = [], []
    for chart in wall['charts']:
        seed = seeds[chart['id']]
        normal = seed.get('normal_camera')
        if chart['cells'] and seed['cells'] and normal is not None and abs(normal[1]) < .67:
            trimmed, ev = patch_wall(chart, seed, fields, size)
            patches = ev['patches']
            contradicted = bool(patches) and all(p.get('opposed',False) for p in patches)
            if candidate == 'planes' or contradicted:
                chart = trimmed
            ev['all_seeds_opposed'] = contradicted
            ev['exported_chart_cells'] = len(chart['cells'])
            evidence.append(ev)
        out.append(chart)
    root = OUT/cohort/candidate
    save(root/'charts'/rel, dict(wall,charts=out))
    save(root/'evidence'/(name+'.json'),dict(id=name,charts=evidence,source_sha256=sha,field_sha256=hashes,
                                            source_chart_sha256=digest(ordinary/rel),whole_chart_sha256=digest(whole/rel)))


def run(job):
    cohort, case, candidate = job
    cv2.setNumThreads(1)
    predict(cohort,case,candidate)
    source = inputs(cohort)[0]
    folder = OUT/cohort/candidate
    mode=(folder/'charts','charts.json',False,1,2.5,.5,False)
    def checked_read(path):
        result = read(path)
        if case['dataset']=='visible-builds-v2' and path==Path(case['frame'])/'frame.json':
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(r for r in audit['frames'] if r['id']==case['id'])
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            return dict(result,framing=proof)
        return result
    with patch.object(paths,'RUN',source), patch.object(assembly,'ASSEMBLY',folder/'assembly'), patch.dict(assembly.MODES,{'connected-atlas':mode}), patch.object(expanded_scores,'read',checked_read):
        assembly.evaluate((case,'sky','connected-atlas'))
    print(cohort,case['id'],flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('cohorts', nargs='+', choices=COHORTS)
    p.add_argument('--candidate', choices=CANDIDATES, default='planes')
    p.add_argument('--workers', type=int, default=5)
    a = p.parse_args()
    jobs = [(cohort, case, a.candidate) for cohort in a.cohorts for case in inputs(cohort)[-1]]
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,jobs))


if __name__ == '__main__':
    main()
