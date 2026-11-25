"""Find thin, downward-facing surfaces with image-native line and shading evidence."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.camera_lines import _runs
from engine.mask_first import _image_depth
from engine.normal_regions import _axes, _field, _local_mask, _patch
from engine.surface_model import canonical


SETTINGS = dict(min_length=15, maximum_slope=.75, minimum_width=18, minimum_area=48,
                minimum_aspect=3.5, direction_cosine=.43, line_radius=2,
                boundary_coverage=.30, shadow_step=6., depth_step=.006,
                max_band_thickness=16, parallel_tolerance=.14,
                pair_relative_darkening=.08, redundant_iou=.65,
                min_flank_disagreement=.20, min_pair_turn=.15, min_turn_gain=.10,
                min_pair_occupied=.88, min_dark_flat_fraction=.65,
                internal_down_cosine=.30, internal_turn_gain=.25,
                existing_down_cosine=.70, existing_purity=.80, absorb_coverage=.65,
                units='Surface-mask pixels, normalized within-image inverse depth; native RGB on 0-255')


def _segments(lines, scale):
    segments = []
    for evidence in lines.values():
        for raw in evidence['segments']:
            a, b = np.asarray(raw, dtype=float).reshape(2,2) / scale
            if b[0] < a[0]:
                a,b = b,a
            width = b[0]-a[0]
            if width >= SETTINGS['min_length'] and abs(b[1]-a[1]) <= SETTINGS['maximum_slope']*width:
                segments.append((a[0],b[0],a[1],(b[1]-a[1])/width))
    return segments


def _tracks(segments, size):
    image = Image.new('L',size)
    draw = ImageDraw.Draw(image)
    for x0,x1,y,slope in segments:
        draw.line((x0,y,x1,y+(x1-x0)*slope),fill=255,width=1)
    return np.asarray(image.filter(ImageFilter.MaxFilter(2*SETTINGS['line_radius']+1)))>0


def _profile(mask, left, top):
    columns = np.nonzero(mask.any(axis=0))[0]
    high = mask[:,columns].argmax(axis=0) + top
    low = mask.shape[0]-mask[::-1,columns].argmax(axis=0) + top
    return columns+left, high, low


def _check(x, high, low, gray, z, toward, labels, tracks, paired):
    # Ends at different x can belong to opposite sides of a support. Work column by column.
    h = gray.shape[0]
    valid = (high>=4)&(low<=h-5)
    x,high,low = x[valid],high[valid],low[valid]
    thickness = low-high
    if len(x)<SETTINGS['minimum_width'] or np.median(thickness)*SETTINGS['minimum_aspect']>len(x):
        return None
    mid = (high+low-1)//2
    above, below = high-3,low+2
    upper,lower = labels[above,x],labels[below,x]
    separation = ((upper!=lower)&(upper!=0)).mean()
    turn = toward[mid,x] - np.maximum(toward[above,x],toward[below,x])
    # A fully merged wall has no change in the two flank IDs. Accept an internal
    # band there only with two independent geometric cues: paired line boundaries
    # and a sustained direction turn inside the band relative to both flanks.
    # At least one strip end must terminate at a different region or an opening.
    # Two mortar lines on an uninterrupted wall can also fool the direction field.
    ends=np.array([0,len(x)-1])
    beyond=np.clip(x[ends]+np.array([-4,4]),0,labels.shape[1]-1)
    terminal=np.any(labels[mid[ends],beyond]!=labels[mid[ends],x[ends]])
    internal = (paired and terminal and np.median(toward[mid,x])>=SETTINGS['internal_down_cosine']
                and np.median(turn)>=SETTINGS['internal_turn_gain'])
    if separation<SETTINGS['min_flank_disagreement'] and not internal:
        return None
    top_cover,bottom_cover = tracks[high,x].mean(),tracks[low-1,x].mean()
    if max(top_cover,bottom_cover)<SETTINGS['boundary_coverage']:
        return None
    shade = gray[above,x]-gray[mid,x]
    decrease = z[above,x]-z[below,x]
    if np.median(shade)<SETTINGS['shadow_step'] or np.median(decrease)<SETTINGS['depth_step']:
        return None
    if paired:
        # A groove in the texture may darken briefly without a new face or depth discontinuity.
        # A sharp drop relative to both flanks, and a dark band over much of the interval,
        # permit this proposal even when the downward direction is weak.
        slope_above = (z[high-4,x]-z[high-2,x])/2
        slope_below = (z[low+2,x]-z[low+4,x])/2
        residual = decrease - (below-above)*(slope_above+slope_below)*.5
        dark = shade > SETTINGS['pair_relative_darkening']*np.maximum(30,gray[above,x])
        inset_high = high + thickness//4
        inset_low = low - 1 - thickness//4
        flat = abs(gray[inset_high,x]-gray[inset_low,x]) < abs(shade)
        if ((dark & flat).mean()<SETTINGS['min_dark_flat_fraction'] or np.median(residual)<SETTINGS['depth_step']
                or np.median(toward[mid,x])<SETTINGS['min_pair_turn']
                or np.median(turn)<SETTINGS['min_turn_gain']):
            return None
    return dict(source='two line segments' if paired else 'line along a downward-facing band',
                upper_line_fraction=float(top_cover),lower_line_fraction=float(bottom_cover),
                different_flank_fraction=float(separation),
                median_brightness_step=float(np.median(shade)),relative_depth_step=float(np.median(decrease)))


def _proposals(segments, labels, toward, gray, z):
    occupied = labels!=0
    size = occupied.shape[::-1]
    tracks = _tracks(segments,size)
    options = []
    # The orientation field can hold a broad, weak downward turn on a soffit even when
    # strict per-pixel direction classification only kept a small fragment.
    for runs in _runs((toward>=SETTINGS['direction_cosine']) & occupied):
        area = sum(b-a for _,a,b in runs)
        if area<SETTINGS['minimum_area']:
            continue
        local,left,top = _local_mask(np.asarray(runs))
        x,high,low = _profile(local,left,top)
        evidence = _check(x,high,low,gray,z,toward,labels,tracks,False)
        if evidence:
            options.append((False,-area,runs,evidence))
    # A top and bottom line can also propose a strip where direction alone has no
    # downward seed at all. Only already occupied pixels can change ownership.
    for i,(a,b,y,slope) in enumerate(segments):
        for c,d,yy,other_slope in segments[i+1:]:
            if abs(slope-other_slope)>SETTINGS['parallel_tolerance']:
                continue
            start,end=max(0,int(np.ceil(max(a,c)))),min(size[0],int(min(b,d)))
            if end-start<SETTINGS['minimum_width']:
                continue
            x=np.arange(start,end)
            one,two=y+(x+.5-a)*slope,yy+(x+.5-c)*other_slope
            if np.any((one<two)!=(one[0]<two[0])):
                continue
            high=np.rint(np.minimum(one,two)).astype(int)
            low=np.rint(np.maximum(one,two)).astype(int)
            thickness=low-high
            if (min(thickness)<2 or max(thickness)>SETTINGS['max_band_thickness']
                    or min(high)<4 or max(low)>size[1]-5):
                continue
            if occupied[(low+high)//2,x].mean()<SETTINGS['min_pair_occupied']:
                continue
            evidence=_check(x,high,low,gray,z,toward,labels,tracks,True)
            if evidence is None:
                continue
            y0,y1=int(min(high)),int(max(low))
            rows=np.arange(y0,y1)[:,None]
            local=(rows>=high)&(rows<low)&occupied[y0:y1,start:end]
            for component in _runs(local):
                area=sum(right-left for _,left,right in component)
                if area>=SETTINGS['minimum_area']:
                    runs=[(v+y0,left+start,right+start) for v,left,right in component]
                    options.append((True,-area,runs,evidence))
    return sorted(options,key=lambda o:o[:2])


def _pieces(labels, parts, options, down):
    original = labels.copy()
    new_parts=[]
    for _,_,runs,evidence in options:
        local,x,y=_local_mask(np.asarray(runs))
        window=np.s_[y:y+local.shape[0],x:x+local.shape[1]]
        old=original[window][local]
        if (labels[window][local]!=old).any():
            continue
        sizes=np.bincount(old,minlength=len(parts)+1)
        # A detection that already belongs largely to one inferred underside is left alone.
        if any(hit and (p['normal_camera'] @ down)>SETTINGS['existing_down_cosine'] and
               (hit/sizes.sum()>=SETTINGS['existing_purity'] or hit/(sizes.sum()+p['support_area']-hit)>=SETTINGS['redundant_iou'])
               for hit,p in zip(sizes[1:],parts)):
            continue
        number=len(parts)+len(new_parts)+1
        # Retain any earlier fragments of this underside if the new outline encloses them.
        include=[i for i,(hit,p) in enumerate(zip(sizes[1:],parts),1)
                 if hit and p['normal_camera'] @ down>SETTINGS['existing_down_cosine']
                 and hit/p['support_area']>=SETTINGS['absorb_coverage']]
        labels[window][local]=number
        for i in include:
            labels[labels==i]=number
        new_parts.append(dict(id=f'underside-{len(new_parts)}',normal_camera=down.tolist(),
                              strip_evidence=evidence))
    return new_parts


def refine(image, parts, lines, normals, depth):
    """Return a partition of the original visible pixels, with proposed underside bands."""
    if not parts:
        return []
    size=tuple(parts[0]['support']['size'])
    scale=np.asarray(image.size)/size
    segments=_segments(lines,scale)
    if not segments:
        return parts
    labels=np.zeros(size[::-1],np.int32)
    for i,p in enumerate(parts,1):
        if tuple(p['support']['size'])!=size:
            raise ValueError('Surface masks must have a common resolution')
        for y,left,right in p['support']['runs']:
            labels[y,left:right]=i
    occupied=labels!=0
    vectors=_field(normals,size)
    down=-_axes(vectors,occupied)[1]
    z=_image_depth(depth,size)
    gray=np.asarray(image.convert('L').resize(size,Image.Resampling.BOX).filter(ImageFilter.GaussianBlur(.7)),np.float32)
    options=_proposals(segments,labels,vectors@down,gray,z)
    additions=_pieces(labels,parts,options,down)
    if not additions:
        return parts
    out=[]
    for i,p in enumerate([*parts,*additions],1):
        mask=labels==i
        area=int(mask.sum())
        if i<=len(parts) and area==p['support_area']:
            out.append(p)
            continue
        if not area:
            continue
        yy,xx=np.nonzero(np.diff(np.pad(mask,((0,0),(1,1))).astype(np.int8),axis=1))
        runs=np.column_stack((yy[::2],xx[::2],xx[1::2])).tolist()
        child=dict(p,support_area=area,support=dict(size=list(size),runs=runs),
                   relative_inverse_depth=float(np.median(z[mask])),geometry_status='mask only')
        child.pop('quad',None)
        chart=_patch(runs,gray[...,None],scale,size,np.asarray(child['normal_camera']))
        if chart:
            child.update(quad=canonical(chart['quad']),geometry_status='mask with sampling chart')
        out.append(child)
    return out
