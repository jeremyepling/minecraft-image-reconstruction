"""Align the narrow boundary of existing surface masks to color transitions in the screenshot."""
import numpy as np
from PIL import Image, ImageFilter


SETTINGS = dict(radius=3, blur=.7, color_step=6., change_cost=1.5, iterations=12,
                minimum_boundary_contrast=10., minimum_contrast_gain=3.,
                units='analysis-mask samples', method='color-guided-boundary')

# Clockwise ring around a pixel. Cardinal neighbors are the odd-numbered entries.
RING = ((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
ADJACENT = (1,3,5,7)


def _touching_components(bits):
    """Number of four-connected neighbor components which actually touch the center."""
    occupied = [bool(bits & (1 << i)) for i in range(8)]
    if all(occupied):
        return 1
    count = 0
    for i in ADJACENT:
        if not occupied[i]:
            continue
        count += 1
        occupied[i] = False
        for step in (-1,1):
            j = (i+step) % 8
            while occupied[j]:
                occupied[j] = False
                j = (j+step) % 8
    return count


# A single flip must preserve components and holes in both its source and destination.
SIMPLE = np.asarray([_touching_components(i)==1 and _touching_components(i^255)==1
                     for i in range(256)])


def _edge_contrast(colors, labels, radius):
    """Mean photographic contrast across the proposed borders in a local window."""
    strength = np.zeros(labels.shape,np.float32)
    length = np.zeros(labels.shape,np.int32)
    for axis in (0,1):
        a,b = ((np.s_[:-1,:],np.s_[1:,:]) if axis==0 else (np.s_[:,:-1],np.s_[:,1:]))
        boundary = labels[a]!=labels[b]
        evidence = boundary*abs(colors[a]-colors[b]).mean(axis=2)
        for side in (a,b):
            length[side] += boundary
            strength[side] += evidence
    def window_sum(values):
        padded = np.pad(values,((radius+1,radius),(radius+1,radius)))
        integral = padded.cumsum(0,dtype=np.float64).cumsum(1)
        k = 2*radius+1
        return integral[k:,k:]+integral[:-k,:-k]-integral[:-k,k:]-integral[k:,:-k]
    return window_sum(strength)/np.maximum(1,window_sum(length))


def _candidate(rgb, labels):
    height,width = labels.shape
    padded = np.pad(labels,1,mode='edge')
    radius = SETTINGS['radius']
    contact = np.zeros(labels.shape, bool)
    for i in ADJACENT:
        dy,dx = RING[i]
        contact |= padded[1+dy:height+1+dy,1+dx:width+1+dx] != labels
    if not contact.any():
        return labels
    # Only refit a narrow collar. Interior labels and the distant background are fixed seeds.
    mask = np.asarray(Image.fromarray(contact).filter(ImageFilter.MaxFilter(2*radius-1)))
    y,x = np.nonzero(mask)
    colors = np.asarray(rgb.filter(ImageFilter.GaussianBlur(SETTINGS['blur'])),np.float32)
    distances = np.zeros(labels.shape,np.float32)
    distances[mask] = 3*radius
    owner = labels.copy()
    neighbors = []
    for i in ADJACENT:
        dy,dx = RING[i]
        ny = (y+dy).clip(0,height-1)
        nx = (x+dx).clip(0,width-1)
        edge = abs(colors[y,x]-colors[ny,nx]).mean(axis=1)
        neighbors.append((ny,nx,1.+edge/SETTINGS['color_step']))
    initial = labels[y,x]
    for _ in range(SETTINGS['iterations']):
        costs = distances[y,x].copy()
        winners = owner[y,x].copy()
        for ny,nx,edge in neighbors:
            candidate = owner[ny,nx]
            cost = distances[ny,nx]+edge+SETTINGS['change_cost']*(candidate!=initial)
            better = (cost<costs-1.e-5) | ((abs(cost-costs)<1.e-5) & (candidate==initial))
            winners[better] = candidate[better]
            costs[better] = cost[better]
        if np.array_equal(costs,distances[y,x]) and np.array_equal(winners,owner[y,x]):
            break
        distances[y,x],owner[y,x] = costs,winners
    old_edge = _edge_contrast(colors,labels,radius+1)
    new_edge = _edge_contrast(colors,owner,radius+1)
    supported = (new_edge>=SETTINGS['minimum_boundary_contrast']) & (
        new_edge>=old_edge+SETTINGS['minimum_contrast_gain'])
    return np.where(supported,owner,labels)


def _retain_topology(labels, target):
    out = np.pad(labels,1,mode='edge')
    y,x = np.nonzero(labels!=target)
    goal = target[y,x]
    # Keep crossings at the image frame fixed. Their continuation outside the image is unknown.
    inside = (y>0)&(x>0)&(y+1<labels.shape[0])&(x+1<labels.shape[1])
    y,x,goal = y[inside]+1,x[inside]+1,goal[inside]
    groups = []
    for iy,ix in ((0,0),(0,1),(1,0),(1,1)):
        group = (y%2==iy)&(x%2==ix)
        groups.append((y[group],x[group],goal[group]))
    for _ in range(SETTINGS['radius']+1):
        changed = False
        # No two centers in a group occur in one another's 3x3 neighborhood.
        for yy,xx,to in groups:
            pending = out[yy,xx]!=to
            if not pending.any():
                continue
            ys,xs,want = yy[pending],xx[pending],to[pending]
            current = out[ys,xs]
            leaving = np.zeros(len(ys),np.uint8)
            joining = np.zeros(len(ys),np.uint8)
            for i,(dy,dx) in enumerate(RING):
                neighbor = out[ys+dy,xs+dx]
                leaving |= ((neighbor==current).astype(np.uint8) << i)
                joining |= ((neighbor==want).astype(np.uint8) << i)
            safe = SIMPLE[leaving] & SIMPLE[joining]
            out[ys[safe],xs[safe]] = want[safe]
            changed |= safe.any()
        if not changed:
            break
    return out[1:-1,1:-1]


def refine(image, parts, version='26.3'):
    """Adjust the predicted support while preserving IDs, normal/depth estimates and sampling charts."""
    if version!='26.3':
        raise ValueError('Surface boundary refinement uses Java 26.3.')
    if not parts:
        return []
    size = tuple(parts[0]['support']['size'])
    labels = np.zeros(size[::-1],np.int32)
    for index,part in enumerate(parts,1):
        if tuple(part['support']['size'])!=size:
            raise ValueError('Surface masks must share the same resolution')
        for y,lo,hi in part['support']['runs']:
            labels[y,lo:hi] = index
    rgb = image.convert('RGB').resize(size,Image.Resampling.BOX)
    result = _retain_topology(labels,_candidate(rgb,labels))
    changed = result!=labels
    touched = set(np.r_[labels[changed],result[changed]].tolist())
    updated = []
    for index,part in enumerate(parts,1):
        if index in touched:
            mask = result==index
            # Horizontal transitions occur in start / end pairs, even on rows containing holes.
            yy,xx = np.nonzero(np.diff(np.pad(mask,((0,0),(1,1))),axis=1))
            runs = np.column_stack((yy[::2],xx[::2],xx[1::2])).tolist()
            part = dict(part,support=dict(size=list(size),runs=runs),support_area=int(mask.sum()))
        updated.append(part)
    return updated
