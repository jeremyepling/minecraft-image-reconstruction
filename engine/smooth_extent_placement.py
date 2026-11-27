"""Recover visible block positions from an unassigned patch of near-uniform material."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from engine.camera_lines import _runs
from engine.pillar_seams import native_mask, orientation
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha, with_outline


METHOD = 'visible-smooth-extent-placement'


def proposals(image, parts, *, common_wall=False):
    native = [_native_alpha(p['support'],image.size) if p.get('support') else None for p in parts]
    boxes = [item[1] for item in native if item is not None]
    if not boxes:
        return []
    left, top = np.maximum(0,np.min(boxes,axis=0)[:2]-12)
    right, bottom = np.minimum(image.size,np.max(boxes,axis=0)[2:]+12)
    rgb = np.asarray(image.convert('RGB'),np.int16)[top:bottom,left:right]
    owners = np.zeros(rgb.shape[:2],np.int32)
    sponsors = defaultdict(set)
    for number,patch in enumerate(native,1):
        if patch is None:
            continue
        alpha,(x,y,r,b) = patch
        mask = np.asarray(alpha,bool)
        window = np.s_[y-top:b-top,x-left:r-left]
        owner = owners[window]
        owner[mask] = np.where(owner[mask]==0,number,-1)
        observed = rgb[window][::3,::3][mask[::3,::3]]
        if len(observed)<80:
            continue
        palette,counts = np.unique(observed,axis=0,return_counts=True)
        for rank in np.argsort(-counts)[:4]:
            if counts[rank]>=16:
                sponsors[tuple(palette[rank])].add(number)
    pending = np.zeros_like(owners)
    unassigned = owners==0
    for color,hosts in sponsors.items():
        close = np.ones(owners.shape,bool)
        for channel,tone in enumerate(color):
            close &= abs(rgb[:,:,channel]-tone)<=3
        # Necessary conditions for any connected candidate below; textured
        # regions rarely have a sufficiently large unassigned swatch.
        if np.count_nonzero(close & unassigned)<64 or np.count_nonzero(close)<500:
            continue
        for component in _runs(close):
            area = sum(b-a for y,a,b in component)
            if area<500 or any(y==0 or y+1==close.shape[0] or a==0 or b==close.shape[1] for y,a,b in component):
                continue
            samples = np.concatenate([owners[y,a:b] for y,a,b in component])
            votes = np.bincount(samples+1,minlength=len(parts)+2)
            host = int(votes[2:].argmax())+1
            # Index zero is an overlap between observed masks; index one is unknown.
            if host not in hosts or votes[1]<64 or votes[host+1]<max(320,.20*area):
                continue
            needed = .985*(area-votes[1])
            if common_wall:
                # Test only color components rejected for divided ownership of a wall.
                # The sponsoring mask remains the owner. Neighbors confirm orientation only.
                normal = parts[host-1].get('normal_camera')
                if votes[host+1]>=needed or normal is None or abs(normal[1])>.5:
                    continue
                aligned = sum(votes[i+2] for i,p in enumerate(parts) if p.get('normal_camera') is not None
                              and np.dot(normal,p['normal_camera'])>=.985)
                if aligned<needed:
                    continue
            elif votes[host+1]<needed:
                continue
            for y,a,b in component:
                active = unassigned[y,a:b]
                old = pending[y,a:b]
                old[active] = np.where((old[active]==0)|(old[active]==host),host,-1)
    regions = []
    for host in np.unique(pending):
        if host<=0:
            continue
        for component in _runs(pending==host):
            area=sum(b-a for y,a,b in component)
            if area>=64:
                regions.append(dict(surface=parts[host-1]['id'],pixels=area,
                                    runs=[(int(y+top),int(a+left),int(b+left)) for y,a,b in component]))
    return regions


def _candidate_parts(parts, size, regions):
    additions = defaultdict(list)
    for r in regions:
        additions[r['surface']].extend(r['runs'])
    changed = []
    for part in parts:
        if part['id'] not in additions:
            changed.append(part)
            continue
        mask = native_mask(part,size)
        for y,a,b in additions[part['id']]:
            mask[y,a:b]=True
        # Preserve the fitted sampling quad. Only this temporary candidate
        # sees the expanded support. Its sampling crop must be regenerated.
        candidate = {k:v for k,v in part.items() if k not in ('outline','source_crop','context_crop','sampling_chart')}
        changed.append(dict(candidate,support_area=int(mask.sum()),support=dict(size=list(size),runs=row_runs(mask))))
    return changed


def supported_additions(image, parts, camera, baseline, candidate, regions):
    from engine.projected_appearance import Scene, STEP
    if candidate<=baseline:
        return set()
    scene = Scene(image,parts,camera,baseline|candidate)
    eligible = np.zeros((3,*scene.shape),bool)
    _,rotation = orientation(camera,image.size)
    normals = {p['id']:p.get('normal_camera') for p in parts}
    for region in regions:
        normal = normals[region['surface']]
        if normal is None:
            continue
        axis=int(np.argmax(abs(rotation.T@normal)))
        for y,a,b in region['runs']:
            if y%STEP==0:
                eligible[axis,y//STEP,(a+STEP-1)//STEP:(b+STEP-1)//STEP]=True
    eligible=eligible.reshape(3,-1)
    additions=set()
    for index,pixels in scene.owned.items():
        cell=scene.cells[index]
        if cell in baseline:
            continue
        n=np.count_nonzero(eligible[scene.axes[pixels],pixels])
        if n>=16 and n>=.6*len(pixels):
            additions.add(cell)
    return additions


def recover(image, version, parts, original, camera, atlas_path):
    projection=np.asarray(camera.get('P') if camera else None,dtype=float)
    if (version!='26.3' or not original or projection.shape!=(3,4)
            or not np.isfinite(projection).all() or np.linalg.matrix_rank(projection[:,:3])!=3):
        return set(), []
    regions=proposals(image,parts)
    if not regions:
        return set(), []
    return recover_from_regions(image,version,parts,original,camera,atlas_path,regions),regions


def recover_from_regions(image, version, parts, original, camera, atlas_path, regions):
    from engine import inset_returns, pavement_opening, texture_guided_count, visible_placement
    from engine.surface_model import _count_regions
    changed=_candidate_parts(parts,image.size,regions)
    # If the new patch suggests an inset or an opening, leave ownership to
    # those dedicated stages with their original visible evidence.
    changed,returns,phase,outlines,_=inset_returns.detect(image,changed,version,camera=camera)
    if returns or phase or outlines or pavement_opening.detect(image,changed,camera):
        return set()
    changed=[with_outline(p) for p in changed]
    with TemporaryDirectory() as dest:
        # This auxiliary geometry fit always uses the visible texture counter.
        # Material sampling in the ordinary build keeps its own phase and assets.
        counts=_count_regions(image,changed,version,texture_guided_count.METHOD,None,Path(dest))
    candidate,_=visible_placement.place(image,version,changed,counts=counts,camera=camera,atlas_path=atlas_path)
    return supported_additions(image,changed,camera,original,set(candidate),regions)
