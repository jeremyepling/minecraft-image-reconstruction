"""Remove source-supported exterior background from upright-build reconstruction."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.pillar_seams import native_mask, orientation
from engine.sky_filter import filter_surface
from engine.projected_appearance import Scene, STEP


def exterior(image):
    """Trace bilateral same-row backdrop colors through connected paths to the frame edges."""
    low = tuple(max(1, side // 3) for side in image.size)
    rgb = np.asarray(image.convert('RGB').resize(low, Image.Resampling.BOX), dtype=np.int16)
    margin = max(1, low[0] // 32)

    def distance(samples):
        refs = np.percentile(samples, [10, 50, 90], axis=1).astype(np.int16)
        return np.minimum.reduce([np.abs(rgb - ref[:, None]).max(axis=2) for ref in refs])

    similar = np.maximum(distance(rgb[:, :margin]), distance(rgb[:, -margin:])) <= 4
    if similar.all():
        return np.zeros(image.size[::-1],bool)
    field = Image.fromarray(np.pad(similar, 1, constant_values=True).astype(np.uint8)).copy()
    ImageDraw.floodfill(field, (0, 0), 2)
    connected = Image.fromarray((np.asarray(field)[1:-1, 1:-1] == 2).astype(np.uint8))
    return np.asarray(connected.resize(image.size, Image.Resampling.NEAREST), dtype=bool)


def supported_camera(camera):
    if not camera or camera.get('P') is None:
        return False
    p = np.asarray(camera['P'],float)
    return p.shape == (3,4) and np.isfinite(p).all() and np.linalg.matrix_rank(p[:,:3]) == 3


def refine(image, parts, camera, background=None):
    """Keep independently observed wall and platform edges while testing horizontal patches."""
    if not parts or not supported_camera(camera):
        return parts, []
    _, rotation = orientation(camera, image.size)
    horizontal = []
    upright = np.zeros(image.size[::-1],bool)
    for p in parts:
        if p.get('normal_camera') is None or not p.get('support'):
            continue
        normal = rotation.T @ p['normal_camera']
        if normal[1] > .8:
            horizontal.append(p)
        elif abs(normal[1]) < .35:
            upright |= native_mask(p, image.size)
    if not horizontal:
        return parts, []
    if background is None:
        background = exterior(image)
    walls = np.asarray(Image.fromarray(upright).filter(ImageFilter.MaxFilter(7)))
    revisions = {}
    for p in horizontal:
        mask = native_mask(p, image.size)
        total = int(mask.sum())
        covered = background & mask
        if total < 8 or covered.sum() < .65 * total:
            continue
        # An adjoining upright face could be the real side of a platform whose top
        # shares the staging floor's shade. Preserve this ambiguous surface.
        if np.count_nonzero(mask & walls) >= 4:
            continue
        clean, log = filter_surface(image, [p], (covered, np.zeros_like(mask)))
        revisions[p['id']] = (clean,dict(id=p['id'],trimmed=int(covered.sum()),remaining=total-int(covered.sum())))
    if not revisions:
        return parts, []
    result = []
    for p in parts:
        if p['id'] in revisions:
            result.extend(revisions[p['id']][0])
        else:
            result.append(p)
    # A corner previously attributed to a backdrop patch is no longer measured.
    for i,p in enumerate(result):
        if 'corner_junctions' in p:
            result[i]=dict(p,corner_junctions=[j for j in p['corner_junctions'] if j['neighbor'] not in revisions])
    return result,[v[1] for v in revisions.values()]


def misplaced(image, parts, camera, cells, background=None):
    """Find elevated blocks whose visible footprint lies wholly on unclaimed exterior pixels."""
    if not cells or not supported_camera(camera):
        return set()
    if background is None:
        background = exterior(image)
    available = background[::STEP,::STEP].copy()
    for part in parts:
        if part.get('support'):
            available &= ~native_mask(part,image.size)[::STEP,::STEP]
    if not available.any():
        return set()
    scene = Scene(image, [], camera, cells)
    available = available.ravel()
    return {scene.cells[i] for i, pixels in scene.owned.items()
            if scene.cells[i][1] > 0 and len(pixels) >= 4 and available[pixels].all()}

def roof_edits(image, original, cleaned, events, status, cells, background=None):
    """Use the adjacent measured roof's own plane to recover faces behind a false shelf."""
    if not events or not cells or not supported_camera(status):
        return set(), set()
    levels={r['id']:r for r in status.get('plane_offsets',[])}
    bad={e['id'] for e in events if e['trimmed'] >= .9*(e['trimmed']+e['remaining'])}
    if not bad:
        return set(),set()
    if background is None:
        background=exterior(image)
    old={p['id']:p for p in original}
    p,rotation=orientation(status,image.size)
    pairs=[]
    for name in sorted(bad):
        patch=native_mask(old[name],image.size)
        rim=np.asarray(Image.fromarray(patch).filter(ImageFilter.MaxFilter(7)))
        for part in cleaned:
            normal=part.get('normal_camera')
            level=levels.get(part['id'])
            if (part['id'] in bad or normal is None or level is None or
                level['axis']!=1 or level['level']<=1 or (rotation.T@normal)[1]<.8):
                continue
            mask=native_mask(part,image.size)
            contact=rim & mask
            if np.count_nonzero(contact)>=16 and np.count_nonzero(contact.any(axis=0))>=8:
                pairs.append((patch,mask,level['level']))
    if len(pairs)!=1:
        return set(),set()
    patch,roof,level=pairs[0]
    eye=-np.linalg.solve(p[:,:3],p[:,3])
    if eye[1]<=level+1:
        return set(),set()
    before=Scene(image,original,status,cells)
    disputed=patch[::STEP,::STEP].ravel()
    support=roof[::STEP,::STEP].ravel()
    backdrop=background[::STEP,::STEP].ravel()
    remove=set()
    for i,visible in before.owned.items():
        cell=before.cells[i]
        if cell[1]!=level:
            continue
        top=visible[before.axes[visible]==1]
        if (len(top)>=4 and disputed[top].mean()>=.45 and backdrop[top].mean()>=.90
                and (backdrop[visible]|support[visible]).mean()>=.96):
            remove.add(cell)
    if not remove:
        return set(),set()
    # Project each image-supported point onto the top of the already measured
    # roof plane. Keep candidates near those released columns.
    y,x=np.nonzero(roof[::STEP,::STEP])
    ray=np.column_stack((x*STEP+.5,y*STEP+.5,np.ones(len(x))))@before.inv.T
    with np.errstate(divide='ignore',invalid='ignore'):
        d=(level-before.eye[1])/ray[:,1]
    ok=(d>1)&(d<150)
    xyz=np.floor(before.eye+ray[ok]*d[ok,None]).astype(int)
    xz,num=np.unique(xyz[:,[0,2]],axis=0,return_counts=True)
    nearby={(x+dx,z+dz) for x,_,z in remove for dx in range(-3,4) for dz in range(-3,4)}
    candidates={(int(x),level-1,int(z)) for (x,z),n in zip(xz,num) if n>=4 and (x,z) in nearby}-cells
    trial=Scene(image,cleaned,status,(cells-remove)|candidates)
    added=set()
    for i,visible in trial.owned.items():
        cell=trial.cells[i]
        if cell in candidates and len(visible)>=4 and ((trial.axes[visible]==1)&support[visible]).mean()>=.90:
            added.add(cell)
    return added,remove
