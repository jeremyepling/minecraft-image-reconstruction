"""Source-only second look at material IDs on photographed cube faces."""
from functools import lru_cache

import numpy as np
from engine.placed_materials import Catalog
from engine.projected_appearance import Scene
from engine.reconstruction import falls_without_support
from engine.thin_atlas_faces import _corr
from benchmark.view_conditioned_atlas import footprint_uv, draw_templates


def face_match(scene, idx, cell, axis, old, atlas, catalog, palette):
    if len(idx) < 20 or old not in catalog.blocks:
        return None
    centers = scene.uv(idx, cell, axis)
    inside = (centers % 16 >= 1) & (centers % 16 < 15) & (centers // 16 >= 1) & (centers // 16 < 15)
    idx = idx[inside]
    if len(idx) < 20:
        return None
    idx = idx[np.linspace(0, len(idx)-1, min(len(idx), 96), dtype=int)]
    y, x = np.divmod(idx, scene.shape[1])
    xy = np.stack((x, y), axis=-1) * scene.step + .5
    uv = footprint_uv(scene.P, xy, cell, axis, samples=3)
    centers = scene.uv(idx, cell, axis)
    pixel = scene.rgb[y, x].astype(np.float32)
    gray = pixel.mean(1)
    if gray.std() < 2:
        return None
    side = ('top' if scene.eye[1] > cell[1]+1 else 'bottom') if axis == 1 else 'wall'
    norm = {'top':1., 'bottom':.5, 'wall':.6 if axis == 0 else .8}[side]
    mean = np.median(pixel / [0.99,1.,1.021], axis=0)
    med = catalog.medians
    gain = np.clip((med @ mean) / np.maximum(1, (med**2).sum(1)), .26, 1.08)
    cost = np.abs(med*gain[:, None] - mean).mean(1)/30 + .16*np.abs(np.log(gain/norm))
    legal = catalog.valid[side]
    old_tex = np.unique(catalog.sprites[legal & (catalog.candidates == catalog.blocks.index(old))])
    if not len(old_tex):
        return None
    candidates = np.unique(np.r_[np.argsort(cost, kind='stable')[:18], old_tex])
    # Both arms compare the identical physical tiles, rotations, pixels and costs.
    maps = catalog.maps[candidates,0]
    static = maps[...,centers]
    view = draw_templates(maps.reshape(-1,16,16),uv).reshape(static.shape)
    # Independent alternating samples validate the orientation selected on the first fold.
    # Use the same orientation on both halves. Narrow fragments need both folds to agree.
    result = []
    for templates in (static, view):
        correlations = np.minimum(_corr(templates[..., ::2],gray[::2]), _corr(templates[..., 1::2],gray[1::2]))
        merit = correlations.max(1) - cost[candidates]
        old_best = max(merit[np.isin(candidates, old_tex)])
        best = int(merit.argmax())
        tex = candidates[best]
        choices = [catalog.blocks[b] for b in catalog.candidates[legal & (catalog.sprites == tex)]]
        chosen = next((b for b in choices if b in palette), choices[0] if choices else old)
        if old in choices or falls_without_support(chosen):
            chosen = old
        win = float(merit[best] - old_best)
        confident = correlations[best].max() >= .46 and win >= .14
        result.append((chosen if confident else old, win if confident else 0., int(tex), int(correlations[best].argmax())))
    return result, (mean,xy,uv,centers,pixel,side), float((uv!=centers[:,None]).mean())


def rerank(image,camera,ids,atlas):
    """Return point-sampled and footprint-sampled IDs and per-cell change evidence."""
    variants = [dict(ids), dict(ids)]
    details = []
    if not ids or camera is None or camera.get('P') is None:
        return *variants, details
    catalog = getattr(atlas, '_placed_catalog', None)
    if catalog is None:
        catalog = atlas._placed_catalog = Catalog(atlas)
    scene = Scene(image.convert('RGB'), [], camera, ids, step=max(1,(max(image.size)+1599)//1600))
    palette = set(ids.values())
    for n,indices in scene.owned.items():
        cell = scene.cells[n]
        old = ids[cell]
        faces = []
        for axis in np.unique(scene.axes[indices]):
            idx = indices[scene.axes[indices]==axis]
            match = face_match(scene,idx,cell,int(axis),old,atlas,catalog,palette)
            if match is not None:
                faces.append((len(idx),int(axis),match))
        if not faces:
            continue
        # The dominant photographed face supplies the second opinion.
        count, axis, (pair, data, mixing) = max(faces,key=lambda row:row[0])
        for k in range(2):
            variants[k][cell] = pair[k][0]
        details.append(dict(xyz=list(cell),axis=axis,pixels=count,footprint_mix=mixing,
                            old=old,static=pair[0][0],view=pair[1][0],
                            static_sprite=pair[0][2:],view_sprite=pair[1][2:]))
    return *variants, details
