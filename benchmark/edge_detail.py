"""Move an existing boundary in a narrow collar at screenshot resolution."""
import numpy as np

from engine.camera_lines import _runs
from engine.plane_consensus import row_runs
from engine.sky_filter import sky_masks
from engine.sky_gap import gap_mask
from engine.surface_boundary_refine import SETTINGS as BASE, _candidate, _retain_topology
from engine.surface_perimeter import _native_alpha


SETTINGS = {name: dict(BASE, radius=radius, units='source pixels') for name,radius in
            [('native2',2), ('native3',3), ('native4',4)]}


SETTINGS['native4-count-guard'] = dict(SETTINGS['native4'], fallback='native2 then original if an existing complete count changes')


def protected_background(labels):
    """Keep empty holes in the original union; exterior can be refined up to the collar."""
    height, width = labels.shape
    holes = np.zeros(labels.shape,bool)
    for spans in _runs(labels==0):
        if all(y>0 and y+1<height and lo>0 and hi<width for y,lo,hi in spans):
            for y,lo,hi in spans:
                holes[y,lo:hi] = True
    return holes


def predict(photo, parts, frame_sha, setting):
    """Use one common label image; freeze overlap pixels already claimed by multiple surfaces."""
    if setting == 'native4-count-guard':
        from engine.masked_count import count
        from engine.surface_dimensions import dimensions
        from engine.surface_sample import sample_surface

        def size(part):
            sample = sample_surface(photo,part)
            if sample is None:
                return [None,None]
            return dimensions(part,sample['quad'],photo.size,count(sample)['counts'])['blocks']

        known = {}
        # Apply a whole-frame fallback so an edge shared by two surfaces always moves together.
        for candidate in ('native4','native2'):
            improved = predict(photo,parts,frame_sha,candidate)
            for index,(before,after) in enumerate(zip(parts,improved)):
                if before is after:
                    continue
                if index not in known:
                    known[index] = size(before)
                if None not in known[index] and known[index] != size(after):
                    break
            else:
                return improved
        return parts
    config = SETTINGS[setting]
    raw = [_native_alpha(p['support'],photo.size) for p in parts]
    extents = [box for native in raw if native is not None for _,box in [native]]
    if not extents:
        return parts
    pad = config['radius']+3
    left=max(0,min(b[0] for b in extents)-pad)
    top=max(0,min(b[1] for b in extents)-pad)
    right=min(photo.width,max(b[2] for b in extents)+pad)
    bottom=min(photo.height,max(b[3] for b in extents)+pad)
    labels=np.zeros((bottom-top,right-left),np.int32)
    for index,native in enumerate(raw,1):
        if native is None:
            continue
        alpha,(x,y,r,b)=native
        region=labels[y-top:b-top,x-left:r-left]
        mask=np.asarray(alpha,bool)
        region[mask]=np.where(region[mask]==0,index,-1)
    target=_candidate(photo.crop((left,top,right,bottom)),labels,config)
    if np.array_equal(labels,target):
        return parts
    blue,cloud=sky_masks(photo)
    sky=blue | cloud | gap_mask(photo,blue,cloud)
    holes=protected_background(labels)
    fixed=(labels==-1)|(target==-1)|((labels==0)&(holes|sky[top:bottom,left:right]))
    target[fixed]=labels[fixed]
    updated=_retain_topology(labels,target,config)
    result=[]
    changed=(labels!=updated)
    touched=set(np.r_[labels[changed],updated[changed]])
    for index,(native,part) in enumerate(zip(raw,parts),1):
        if native is None or index not in touched:
            result.append(part)
            continue
        alpha,(x,y,r,b)=native
        mask=updated==index
        # Overlapping masks retain their original owners.
        mask[y-top:b-top,x-left:r-left] |= np.asarray(alpha,bool) & (labels[y-top:b-top,x-left:r-left]==-1)
        cleaned={k:v for k,v in part.items() if k not in ('outline','source_crop','context_crop','sampling_chart')}
        runs=(np.asarray(row_runs(mask))+[top,left,left]).tolist()
        cleaned.update(support=dict(runs=runs,size=list(photo.size)),support_area=int(mask.sum()))
        result.append(cleaned)
    return result
