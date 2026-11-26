"""Measure image evidence for block coordinates already proposed by visible placement."""
import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.visible_raster import SAMPLE_STRIDE, observed_support


FEATURES = (
    "visible_fraction", "matching_mask", "visible_matching_mask", "outside_building",
    "conflicting_normal", "face_count", "aligned_faces", "best_face_support", "best_atlas",
    "best_supported_atlas", "plane_atlas", "plane_median_atlas", "coplanar_neighbors",
    "matching_neighbors", "surface_ground_contact", "normal_strength", "horizontal_face",
    "edge_contrast", "visible_samples", "atlas_separation",
)


def _texture_agreement(samples, supported, atlas):
    """Correlate each visible face's sampled light/dark pattern with vanilla art."""
    scores = np.zeros(len(samples))
    margin = np.zeros(len(samples))
    if atlas is None:
        return scores, margin
    for i in np.flatnonzero(supported.sum(axis=1) >= 96):
        keep = supported[i]
        colors = samples[i,keep]
        mean = colors.mean(axis=0)
        gain = np.clip((atlas.means @ mean) / np.maximum(1,(atlas.means**2).sum(1)),.3,1.15)
        err = np.abs(atlas.means*gain[:,None] - mean).mean(1) + 2*np.abs(np.log(gain/.8))
        indices = np.argpartition(err, 24)[:24]
        # Subtract the mean over the observed area of each specific template.
        templates = atlas.patterns[indices].reshape(-1,256)[:,keep]
        templates = templates - templates.mean(1,keepdims=True)
        observed = colors.mean(1)
        observed -= observed.mean()
        if observed.std() < 1.:
            continue
        value = (templates @ observed) / np.maximum(1.,np.linalg.norm(templates,axis=1)*np.linalg.norm(observed))
        by_texture = value.reshape(len(indices),8).max(1)
        best = np.partition(by_texture,-2)[-2:]
        scores[i] = best[1]
        margin[i] = best[1] - best[0]
    return scores,margin


def cell_features(image, parts, camera, cells, *, atlas_path=None, masks=None, support_only=False):
    """Return one feature row per sorted cell. RGB and predicted geometry are the only inputs."""
    xyz = np.array(sorted(cells),dtype=int).reshape(-1,3)
    if len(xyz) == 0:
        return xyz,np.empty((0,len(FEATURES))),False
    from engine.visible_placement import _frontmost_cells
    projection = np.array(camera,dtype=float)
    projection /= np.linalg.norm(projection[2,:3])
    eye = np.linalg.solve(projection[:,:3],-projection[:,3])
    shape = tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in image.size[::-1])
    if masks is None:
        masks = [observed_support(part['support'],shape,image.size) if part.get('support')
                 else np.zeros(shape,dtype=bool) for part in parts]
    owner = _frontmost_cells(list(map(tuple,xyz)),projection,eye,shape)
    union = np.logical_or.reduce(masks)
    focal = np.linalg.norm(projection[0,:3]-image.width/2*projection[2,:3])
    rotation = np.linalg.solve(intrinsic(image.size,focal),projection[:,:3])
    atlas = None if support_only else _atlas('26.3',atlas_path)
    rgb = np.asarray(image.convert('RGB'))
    assigned = np.zeros((3,*shape),dtype=bool)
    ground = np.zeros((3,*shape),dtype=float)
    strength = np.zeros_like(ground)
    bottom = union & (np.arange(shape[0])[:,None] == np.where(union,np.arange(shape[0])[:,None],-1).max(0))
    for part,mask in zip(parts,masks,strict=True):
        n = part.get('normal_camera')
        if n is None or not mask.any():
            continue
        alignment = np.abs(rotation.T @ n)
        axis = int(alignment.argmax())
        assigned[axis] |= mask
        contact = (mask & bottom).sum() / max(1,mask.any(0).sum())
        ground[axis,mask] = np.maximum(ground[axis,mask],contact)
        strength[axis,mask] = np.maximum(strength[axis,mask],alignment[axis])

    u,v = np.meshgrid((np.arange(16)+.5)/16,(np.arange(16)+.5)/16)
    views = []
    indices = np.arange(len(xyz))[:,None]
    for axis in range(3):
        orient = (eye[axis] >= xyz[:,axis]+1)
        level = xyz[:,axis]+orient
        facing = orient | (eye[axis] <= xyz[:,axis])
        offsets = np.zeros((256,3))
        free = [a for a in range(3) if a!=axis]
        offsets[:,free] = np.column_stack((u.ravel(),v.ravel()))
        world = xyz[:,None,:]+offsets
        world[:,:,axis] = level[:,None]
        q = world @ projection[:,:3].T+projection[:,3]
        xy = q[:,:,:2]/np.maximum(q[:,:,2,None],1.e-6)
        pix = np.floor(xy).astype(int)
        valid = facing[:,None] & (q[:,:,2]>1.e-6) & (pix>=0).all(2) & (pix<image.size).all(2)
        px,py = np.clip(pix,[0,0],np.array(image.size)-1).transpose(2,0,1)
        sx,sy = px//SAMPLE_STRIDE,py//SAMPLE_STRIDE
        visible = valid & (owner[sy,sx]==indices)
        match = valid & assigned[axis,sy,sx]
        support = visible & match
        if support_only:
            views.append(support.mean(1))
            continue
        a,b=xy[:,15]-xy[:,0],xy[:,240]-xy[:,0]
        footprint = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]) / SAMPLE_STRIDE**2
        samples = rgb[py,px].astype(np.float32)
        art,margin = _texture_agreement(samples,support,atlas)
        gray = samples.mean(2).reshape(-1,16,16)
        contrast = np.abs(np.diff(gray,axis=1)).mean((1,2)) + np.abs(np.diff(gray,axis=2)).mean((1,2))
        plane_art = np.zeros(len(xyz))
        med_art = np.zeros(len(xyz))
        for l in np.unique(level):
            group = (level==l) & (support.sum(1)>=96)
            if group.any():
                plane_art[level==l] = np.mean(art[group]>.65)
                med_art[level==l] = np.median(art[group])
        visible_count = np.maximum(1,visible.sum(1))
        views.append(dict(visible=visible.mean(1),match=match.mean(1),supported=support.mean(1),
            conflict=(visible & union[sy,sx] & ~match).sum(1)/visible_count,
            background=(visible & ~union[sy,sx]).sum(1)/visible_count,
            contact=(ground[axis,sy,sx]*support).sum(1)/np.maximum(1,support.sum(1)),
            strength=(strength[axis,sy,sx]*support).sum(1)/np.maximum(1,support.sum(1)),
            projected=footprint,art=art,margin=margin,plane=plane_art,median=med_art,contrast=contrast))
    if support_only:
        return xyz,np.maximum.reduce(views),False
    support=np.stack([view['supported'] for view in views],axis=1)
    dominant=np.argmax(support,axis=1)
    pick=lambda key: np.stack([view[key] for view in views],axis=1)[np.arange(len(xyz)),dominant]
    visible=np.stack([view['visible'] for view in views],axis=1)
    art=np.stack([view['art'] for view in views],axis=1)
    lookup = {tuple(cell):i for i,cell in enumerate(xyz)}
    neighbors = np.zeros(len(xyz))
    matching = np.zeros(len(xyz))
    for i,(cell,axis) in enumerate(zip(xyz,dominant,strict=True)):
        for free in (a for a in range(3) if a!=axis):
            for step in (-1,1):
                delta=cell.copy()
                delta[free]+=step
                j=lookup.get(tuple(delta))
                if j is not None:
                    neighbors[i]+=1
                    matching[i]+=int(art[j,axis]>.65)
    values=np.column_stack([pick('visible'),pick('match'),pick('supported'),pick('background'),pick('conflict'),
            (visible>.25).sum(1),(support>.4).sum(1),support.max(1),art.max(1),pick('art'),pick('plane'),
            pick('median'),neighbors,matching,pick('contact'),pick('strength'),dominant==1,pick('contrast'),
            pick('projected')*pick('visible'),pick('margin')])
    return xyz,np.nan_to_num(values),atlas is not None
