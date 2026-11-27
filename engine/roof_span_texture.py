"""Measure the integer extents of photographed roof strips against projected atlas texels."""
from engine import floating_grid_policy as floating
from collections import defaultdict

import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.thin_atlas_faces import StripEvidence, _corr, _match
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _sample_face


def strip_lines(prior):
    """Both a roof fragment and a vertical contact can nominate its visible span."""
    layers = defaultdict(set)
    for x, y, z in prior:
        if (floating.active() or y > 0) and (x, y + 1, z) not in prior:
            layers[y].add((x, z))
            if (x, y-1, z) in prior:
                layers[y+1].add((x, z))
    # The window controls the search only. Each emitted face needs a native pixel witness.
    faces = defaultdict(set)
    for y, anchors in layers.items():
        if len(anchors) < 2:
            continue
        for x, z in anchors:
            for direction in (0, 2):
                for offset in range(-9, 10):
                    cell = (x + (offset if direction == 0 else 0), y, z + (offset if direction == 2 else 0))
                    if cell not in prior:
                        faces[1, y].add(cell)
                        axis = 2 - direction
                        faces[axis, cell[axis]].add(cell)
    return faces


def visible_candidates(ctx, lines):
    # Intersect native photo rays with each strip plane. This both bounds the
    # extent by the observed outline and avoids testing every cube in a box.
    y, x = np.nonzero(ctx.union[::2, ::2])
    x, y = x*2, y*2
    rays = ctx._rays(x, y)
    for (axis, lower), cells in lines.items():
        if lower <= ctx.eye[axis] <= lower+1:
            continue
        plane = lower + int(ctx.eye[axis] > lower+1)
        depth = rays[:, axis] / (plane-ctx.eye[axis])
        votes = (depth > ctx.depth[y,x] + 1e-8) & (ctx.masks[axis,y,x] | (ctx.axes[y,x]<0))
        if not votes.any():
            continue
        xyz = ctx.eye + rays[votes]/depth[votes,None]
        coords = np.floor(xyz).astype(int)
        coords[:,axis] = lower
        pts, counts = np.unique(coords,axis=0,return_counts=True)
        for point in pts[counts>=4].tolist():
            cell=tuple(point)
            if cell in cells:
                yield cell,axis


def measure(ctx, cell, axis, joint=None, *, palette=None):
    sign = 1 if ctx.eye[axis]>cell[axis]+1 else -1
    hit = _sample_face(ctx.native, cell, axis, sign, ctx.shape)
    if hit is None:
        return None
    box, face, depth, *_ = hit
    front = face & (depth > ctx.depth[box]+1e-8)
    if joint is not None:
        owner, axes, ids = joint
        front &= (owner[box]==ids[cell]) & (axes[box]==axis)
    mask = front & ctx.union[box]
    area, n = int(front.sum()), int(mask.sum())
    if n < 18 or n < .84 * area:
        return None
    y,x = np.nonzero(mask)
    if len(x)>180:
        take = np.linspace(0,len(x)-1,180,dtype=int)
        x,y = x[take],y[take]
    x,y = x+box[1].start,y+box[0].start
    colors = ctx.rgb[y,x]
    if colors.mean(1).std() < .75:
        return None
    rays = ctx._rays(x, y)
    free = [i for i in range(3) if i != axis]
    slope = rays[:,free]/rays[:,[axis]]
    uv = ctx.eye[free] + (cell[axis]+int(sign>0)-ctx.eye[axis])*slope
    # Neighboring photographed materials shortlist sprites. Individual blocks
    # may select different sprites across material transitions on the same roof.
    if palette is None:
        palette = list(ctx.anchors)
    match = _match(np.floor(uv*16).astype(int),colors,ctx.atlas,ctx.maps,palette)
    if not match or match['texels']<10 or match['split']<.52 or match['color']>12:
        return None
    template = ctx.maps[match['sprite'],match['mip'],match['turn']]
    gray = colors.mean(1)
    def correlation(coordinates):
        bins = np.floor(16*coordinates).astype(int)%16
        return float(_corr(template[bins[:,1]*16+bins[:,0]],gray))
    margin = correlation(uv)-max(correlation(uv+s*slope) for s in (-1,1))
    aligned = float(ctx.masks[axis,y,x].mean())
    return dict(match,depth=margin,n=n,inside=n/area,oriented=aligned)


def continuous(rows, prior, mode):
    # Recover measured runs along either horizontal grid direction. A run can
    # terminate at a texture change or the photographed outline. Missing
    # witnesses always break the run, even between two measured contacts.
    score, gap = { 'loose':(.62,.09), 'strict':(.75,.16), 'guarded':(.62,.09),
                   'conservative':(.75,.16) }[mode]
    cells = {cell for (cell,_),r in rows.items()
             if r['split']>=score and r['depth']>=gap and r['phase']>=.075
             and r['color']<=9 and (r['oriented']>.55 or r['split']>=.84)}
    placed=set()
    strong={cell for (cell,_),r in rows.items() if r['split']>=.78 and r['color']<=8
            and r['depth']>=.18 and r['phase']>=.10}
    for x,y,z in cells:
        for dx,dz in ((1,0),(0,1)):
            if (x-dx,y,z-dz) in cells:
                continue
            run=[]
            px,pz=x,z
            while (px,y,pz) in cells:
                run.append((px,y,pz))
                px,pz=px+dx,pz+dz
            start, end = (x-dx,y,z-dz) in prior, (px,y,pz) in prior
            anchor = start or end
            contact = any((rx,y-1,rz) in prior for rx,_,rz in run)
            if len(run)>=2 and (anchor or contact):
                placed.update(run)
            elif mode=='guarded' and len(run)==1 and anchor and (start and end or run[0] in strong):
                placed.update(run)
    return placed


def observations(image,parts,camera,prior,atlas_path=None):
    if not prior or camera.get('P') is None or not any(p.get('support') for p in parts):
        return None, {}
    atlas = _atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256') != ATLAS_SHA256:
        return None, {}
    ctx=StripEvidence(image,parts,camera,prior,atlas)
    rows={}
    for cell,axis in visible_candidates(ctx,strip_lines(prior)):
        row=measure(ctx,cell,axis)
        if row:
            rows[cell,axis]=row
    return ctx, rows


def proposed(ctx,rows,prior,mode):
    selected=continuous(rows,prior,mode)
    if not selected:
        return prior
    joint=ctx.joint(selected)
    checked={}
    for (c,axis) in rows:
        if c in selected:
            row=measure(ctx,c,axis,joint)
            if row:
                checked[c,axis]=row
    final=continuous(checked,prior,mode)
    if not final:
        return prior
    if mode in ('guarded','conservative'):
        # Check *all* projected faces, including vertical ends of accepted top
        # strips. Every newly intercepted ray must stay within the photographed
        # silhouette, allowing a one-pixel rasterization discrepancy at edges.
        padded=np.pad(ctx.union,1)
        h,w=ctx.shape
        inside=np.logical_or.reduce([padded[dy:dy+h,dx:dx+w] for dy in range(3) for dx in range(3)])
        empty=ctx.depth==0
        while final:
            owner,_,ids=ctx.joint(final)
            rejected={i for i in np.unique(owner[empty & ~inside]) if i>=0}
            kept={c for c in final if ids[c] not in rejected}
            if kept==final:
                break
            final=kept
    return _keep_prior_visible(ctx.size,ctx.P,ctx.eye,prior,final)


def recover(image,parts,counts,camera,prior,*,atlas_path=None,mode='guarded'):
    ctx,rows=observations(image,parts,camera,prior,atlas_path)
    return proposed(ctx,rows,prior,mode),dict(measured_faces=len(rows))


def conservative(image,parts,counts,camera,prior,*,atlas_path=None):
    return recover(image,parts,counts,camera,prior,atlas_path=atlas_path,mode='conservative')
