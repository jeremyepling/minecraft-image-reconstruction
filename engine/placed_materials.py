"""Assign vanilla block IDs to the final, first-hit visible placements."""
import numpy as np

from engine.projected_appearance import Scene
from engine.reconstruction import falls_without_support, stabilize_placements
from engine.thin_atlas_faces import _corr, _mips

FALLBACK = 'smooth_stone'
METHODS = ('face', 'palette', 'blend', 'auto')


class Catalog:
    def __init__(self, atlas):
        quantiles = np.percentile(atlas.tiles.reshape(-1, 256, 3), (15, 50, 85), axis=1)
        self.medians = quantiles[1].astype(np.float32)
        self.spread = ((quantiles[2] - quantiles[0]) * .5).astype(np.float32)
        self.maps = _mips(atlas)
        gray = atlas.tiles.astype(np.float32).mean(3)
        self.gradient = {lag:np.stack([abs(gray-np.roll(gray, lag, k)).mean((1,2)) for k in (1,2)])
                         for lag in (1,2,4,8)}
        flat = atlas.tiles.reshape(-1,256,3).astype(np.float32)
        self.chroma = np.percentile((flat[:,:,[0,2]]+2)/(flat[:,:,[1]]+2), (20,50,80),axis=1)
        self.blocks = sorted({name.removeprefix('minecraft:') for ids in atlas.ids for name in ids})
        index = {name: i for i, name in enumerate(self.blocks)}
        self.sprites, self.candidates, self.tie = [], [], []
        for texture, choices in enumerate(atlas.ids):
            for order, name in enumerate(choices):
                self.sprites.append(texture)
                self.candidates.append(index[name.removeprefix('minecraft:')])
                self.tie.append(.04 * min(order, 10))
        self.sprites = np.asarray(self.sprites)
        self.candidates = np.asarray(self.candidates)
        self.tie = np.asarray(self.tie)
        self.valid = {}
        for face, directions in {'wall':('north','south','east','west'), 'top':('up',), 'bottom':('down',)}.items():
            choices = getattr(atlas, 'face_choices', None)
            self.valid[face] = np.asarray([
                not choices or any(self.blocks[b] in choices[d].get(atlas.labels[t], ()) for d in directions)
                for t,b in zip(self.sprites,self.candidates)], dtype=bool)

    def face(self, pixels, uv, normal_gain, side):
        # Interior texels tolerate small camera and outline errors at boundaries.
        inner = (uv % 16 >= 2) & (uv % 16 < 14) & (uv // 16 >= 2) & (uv // 16 < 14)
        if inner.sum() >= max(5, len(uv) // 4):
            pixels, uv = pixels[inner], uv[inner]
        if len(uv) > 256:
            chosen = np.linspace(0, len(uv)-1, 256, dtype=int)
            pixels, uv = pixels[chosen], uv[chosen]
        # Vanilla daylight has a slight blue cast in these source captures.
        pixels = pixels / np.array([.99, 1., 1.021], dtype=np.float32)
        q = np.percentile(pixels, (15, 50, 85), axis=0)
        median, spread = q[1], .5 * (q[2] - q[0])
        gain = np.clip((self.medians * median).sum(1) / np.maximum(1, (self.medians**2).sum(1)),
                       .26, 1.08)
        residual = np.abs(self.medians * gain[:, None] - median).mean(1)
        # Shadows are common underneath rims. Cap their directional penalty.
        shade = abs(np.log(gain / normal_gain))
        contrast = np.abs(np.log((self.spread * gain[:, None] + 1.6)/(spread+1.6))).mean(1)
        loss = residual * 1.7 + 4.0 * contrast + 8.0 * shade
        chroma = np.percentile((pixels[:,[0,2]]+2)/(pixels[:,[1]]+2), (20,50,80),axis=0)
        loss += 35 * np.abs(np.log(self.chroma/chroma[:,None])).mean((0,2))
        if len(uv) >= 8:
            occupied, group, counts = np.unique(uv, return_inverse=True, return_counts=True)
            if len(occupied) >= 8:
                gray = (np.bincount(group, weights=pixels.mean(1)) / counts).astype(np.float32)
                indexed = np.full(256, np.nan)
                indexed[occupied] = gray
                for lag, weight in ((1,3.8),(2,1.8),(4,1.8),(8,1.5)):
                    gradients = []
                    for across in ((occupied%16+lag)%16+16*(occupied//16), (occupied+lag*16)%256):
                        diffs = abs(indexed[across]-gray)
                        valid = diffs[np.isfinite(diffs)]
                        if len(valid) >= 5:
                            gradients.append(np.mean(np.minimum(valid, np.percentile(valid, 90))))
                    if len(gradients) == 2:
                        measured = np.sort(gradients)
                        predicted = np.sort(self.gradient[lag],axis=0) * gain
                        loss += weight * abs(np.log((predicted+1.2)/(measured[:,None]+1.2))).mean(0)
                short = np.argsort(loss, kind='stable')[:18]
                corr = _corr(self.maps[short][..., occupied], gray).max(axis=(1, 2))
                strength = min(1., (len(occupied)-4)/32)
                if len(occupied) >= 18 and gray.std() > .7:
                    coords = np.stack((occupied%16, occupied//16, np.ones(len(occupied))), axis=1)
                    highpass = gray - coords @ np.linalg.lstsq(coords,gray,rcond=None)[0]
                    templates = self.maps[short,0]
                    for dy,dx in ((-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1),(0,0)):
                        shifted = (occupied//16+dy)%16*16 + (occupied+dx)%16
                        score = _corr(templates[...,shifted], highpass).max(1)
                        corr = np.maximum(corr, score - .018*(abs(dx)+abs(dy)))
                loss[short] -= 19 * strength * np.maximum(0., corr-.32)
        block_loss = np.full(len(self.blocks), 1000., dtype=np.float32)
        valid = self.valid[side]
        np.minimum.at(block_loss, self.candidates[valid], loss[self.sprites[valid]] + self.tie[valid])
        return block_loss


def _fallback(cells):
    return {c:FALLBACK for c in cells}, {c:'fallback' for c in cells}


def _pool(records, catalog):
    """Combine registered texels across similar-coloured cubes on the same plane."""
    groups = {}
    for rec in records:
        cell, axis, _, side, pixels, uv, _, _ = rec
        median = np.median(pixels,axis=0)
        key = (axis, cell[axis], side)
        group_list = groups.setdefault(key, [])
        for group in group_list:
            anchor = group[0]
            gain = np.clip(median @ anchor / max(1.,anchor @ anchor), .82,1.20)
            if np.mean(abs(median-anchor*gain)) < 3.2:
                break
        else:
            group = (median, [])
            group_list.append(group)
        group[1].append((rec, median))
    for group_list in groups.values():
        for anchor, members in group_list:
            if len(members) < 3:
                continue
            # Give each cube comparable weight. Register first, then average
            # repeated texels; this allows different slivers to complete a tile.
            sums, counts = np.zeros((256,3)), np.zeros(256)
            for rec, median in members:
                uv, colors = rec[5], rec[4]
                occupied, inverse, n = np.unique(uv,return_inverse=True,return_counts=True)
                for channel in range(3):
                    centered = colors[:,channel]-median[channel]+anchor[channel]
                    sums[occupied,channel] += np.bincount(inverse,weights=centered)/n
                counts[occupied] += 1
            seen = np.flatnonzero(counts>0)
            if len(seen) < 22:
                continue
            _,_,gain,side,*_ = members[0][0]
            shared = catalog.face(sums[seen]/counts[seen,None],seen,gain,side)
            for rec,_ in members:
                fraction = .65 if len(np.unique(rec[5])) < 45 else .40
                rec[6] = (1-fraction)*rec[6] + fraction*shared


def _stabilize(ids, evidence, costs, observed, catalog):
    stable = np.array([not falls_without_support(name) for name in catalog.blocks])
    replacements = {}
    for i,cell in enumerate(observed):
        if falls_without_support(ids[cell]):
            alternate = np.argmin(np.where(stable, costs[i], np.inf))
            replacements[cell] = catalog.blocks[alternate] if stable.any() else FALLBACK
    for change in stabilize_placements(ids,replacements):
        pos = tuple(change[d] for d in ('x','y','z'))
        evidence[pos] = 'source-atlas:unsupported_gravity_block:' + change['predicted']


def _refine(loss, observed, reliability=1.):
    # Shared top textures give several block IDs equal support. Distribute their
    # votes so a handful of distinctive wall faces can resolve the actual ID.
    refined = loss
    for _ in range(3):
        support = np.exp(-(refined-refined.min(1,keepdims=True))/1.35)
        support /= support.sum(1,keepdims=True)
        refined = loss - 3.2 * np.log1p((support * reliability).sum(0))
    # Nearby faces with similar evidence often form one building material.
    # Keep the local vote bounded so a one-block trim can retain a clear match.
    winners = refined.argmin(1)
    lookup = {c:i for i,c in enumerate(observed)}
    local = refined.copy()
    for i,(x,y,z) in enumerate(observed):
        nearby = [lookup[c] for c in ((x-1,y,z),(x+1,y,z),(x,y-1,z),(x,y+1,z),(x,y,z-1),(x,y,z+1))
                  if c in lookup]
        if nearby:
            for choice, n in zip(*np.unique(winners[nearby], return_counts=True)):
                local[i,choice] -= 2.4 * np.log1p(n)
    return local


def assign_many(image, camera, cells, atlas, *, methods=METHODS):
    """Compare matchers on identical cubes, pixels and camera without repeated rasterization."""
    if not set(methods).issubset(METHODS):
        raise ValueError(f'Unknown placement material method in {methods}')
    cells = sorted({tuple(map(int, c)) for c in cells})
    if not cells or atlas is None or camera is None or camera.get('P') is None:
        return {m:_fallback(cells) for m in methods}
    matrix = np.asarray(camera['P'], dtype=float)
    if matrix.shape != (3,4) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix[:,:3]) < 3:
        return {m:_fallback(cells) for m in methods}
    scene = Scene(image.convert('RGB'), [], camera, cells, step=max(1, (max(image.size)+1599)//1600))
    catalog = getattr(atlas, '_placed_catalog', None)
    if catalog is None:
        catalog = atlas._placed_catalog = Catalog(atlas)
    rgb = scene.rgb.reshape(-1,3).astype(np.float32)
    grouped = {}
    records = []
    individual = {}
    reliability = {}
    for number, indices in scene.owned.items():
        cell = cells[number]
        for axis in np.unique(scene.axes[indices]):
            idx = indices[scene.axes[indices] == axis]
            if len(idx) < 2:
                continue
            side = ('top' if scene.eye[1] > cell[1]+1 else 'bottom') if axis == 1 else 'wall'
            direction = {'top':1., 'bottom':.5, 'wall':(.60 if axis == 0 else .80)}[side]
            uv = scene.uv(idx, cell, axis)
            pixels = rgb[idx]
            weight = min(40.,len(idx)**.65)
            rec = [cell,axis,direction,side,pixels,uv,catalog.face(pixels,uv,direction,side),weight]
            y,x = np.divmod(idx,scene.shape[1])
            sparse = (x%2 == 0) & (y%2 == 0)
            if sparse.sum() >= 2:
                sparse_cost = catalog.face(pixels[sparse], uv[sparse], direction, side)
                individual.setdefault(cell,[]).append((sparse_cost,min(24., sparse.sum()**.65)))
            reliability[cell] = max(reliability.get(cell,0), len(np.unique(uv)))
            records.append(rec)
            grouped.setdefault(cell,[]).append(rec)
    observed = list(grouped)
    if not observed:
        return {m:_fallback(cells) for m in methods}
    _pool(records,catalog)
    loss = np.array([np.average([r[6] for r in group],axis=0,weights=[r[7] for r in group])
                     for group in grouped.values()])
    first = np.argmin(loss, axis=1)
    # A source image supplies its own palette. Shrink uncertain guesses toward
    # IDs independently supported on other cubes, while preserving distinct accents.
    counts = np.bincount(first, minlength=len(catalog.blocks))
    prior = np.log1p(counts)
    adjusted = loss - 2.5 * prior
    sparse_loss = []
    for i,cell in enumerate(observed):
        faces = individual.get(cell)
        sparse_loss.append(np.average([r[0] for r in faces], axis=0, weights=[r[1] for r in faces])
                           if faces else loss[i])
    mixed = (loss + np.array(sparse_loss)) * .5
    precision = np.array([np.clip((reliability[c]/140)**1.5,.08,1.) for c in observed])[:,None]
    costs = {'face':loss,'palette':adjusted,'blend':_refine(mixed,observed),
             'auto':_refine(mixed,observed, precision / precision.mean())}
    selections = {method:costs[method].argmin(1) for method in methods}
    result = {}
    for method in methods:
        ids, evidence = _fallback(cells)
        for cell, choice in zip(observed, selections[method]):
            ids[cell] = catalog.blocks[choice]
            evidence[cell] = 'source-atlas'
        _stabilize(ids, evidence, costs[method], observed, catalog)
        result[method] = ids, evidence
    return result


def assign(image, camera, cells, atlas, *, method='auto'):
    """Return bare Minecraft IDs and evidence status, each keyed by local xyz."""
    return assign_many(image, camera, cells, atlas, methods=(method,))[method]

