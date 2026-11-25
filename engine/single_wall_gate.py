"""Allow one depth-supported MobileSAM shape to separate a vertical wall."""
import numpy as np
from PIL import Image

from benchmark.mask_metrics import footprint
from benchmark.mobile_wall_split import children


STRIDE = 4
SETTINGS = dict(max_vertical_normal_y=.5, min_parent_share=.035, min_side_area=24,
                min_containment=.55, max_parent_share=.92, min_sample_pairs=6,
                near_distance=0, far_distance=2, min_step=.001, min_agreement=.70,
                min_seed_samples=12)


def depth_grid(field, size):
    """Read the same native pixel centers as the frozen MobileSAM numerical masks."""
    values=np.asarray(field,dtype=np.float32)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Expected one finite inverse-depth field')
    w,h=size
    if values.shape!=(h,w):
        values=np.asarray(Image.fromarray(values).resize(size,Image.Resampling.BILINEAR))
    return values[2:2+h//STRIDE*STRIDE:STRIDE,2:2+w//STRIDE*STRIDE:STRIDE].copy()


def sample_pairs(side,parent,near,far):
    """Take four points on a straight run crossing the internal mask boundary exactly once."""
    h,w=side.shape
    samples=[]
    for dy,dx in ((0,1),(1,0)):
        yy,xx=np.nonzero((side[:h-dy,:w-dx]!=side[dy:,dx:]) &
                         parent[:h-dy,:w-dx] & parent[dy:,dx:])
        # Start at the last sample inside the proposal, point toward its remainder.
        sign=np.where(side[yy,xx],1,-1)
        yy=yy+dy*(sign<0)
        xx=xx+dx*(sign<0)
        steps=np.arange(-far,far+2)
        y=yy[:,None]+dy*sign[:,None]*steps
        x=xx[:,None]+dx*sign[:,None]*steps
        valid=((y>=0)&(x>=0)&(y<h)&(x<w)).all(axis=1)
        y,x=y[valid],x[valid]
        valid=(parent[y,x] & (side[y,x]==(steps<=0))).all(axis=1)
        idx=np.asarray([-far,-near,1+near,1+far])+far
        points=np.stack((y[valid][:,idx],x[valid][:,idx]),axis=2).reshape(-1,8)
        samples.extend(points.tolist())
    return samples


def prepare(image,regions,masks,records,selected):
    """Freeze eligible cuts and sample locations before opening either depth field."""
    w,h=image.size
    shape=(h//STRIDE,w//STRIDE)
    proposals=[]
    for part in regions:
        if abs(part['normal_camera'][1])>SETTINGS['max_vertical_normal_y']:
            continue
        parent=footprint(part,shape,image.size,True)
        area=int(parent.sum())
        small=max(SETTINGS['min_side_area'],area*SETTINGS['min_parent_share'])
        for i in selected:
            shape_mask=masks[i]
            cut=parent & shape_mask
            inside=int(cut.sum())
            if min(inside,area-inside)<small or inside>area*SETTINGS['max_parent_share']:
                continue
            containment=inside/records[i]['area']
            if containment<SETTINGS['min_containment']:
                continue
            samples=sample_pairs(cut,parent,SETTINGS['near_distance'],SETTINGS['far_distance'])
            if len(samples)>=SETTINGS['min_sample_pairs']:
                proposals.append(dict(parent=part['id'],proposal=int(i),samples=samples,
                                      inside_samples=inside,outside_samples=area-inside,containment=containment))
    return proposals


def votes(inventory,field):
    """Remove a locally linear slope; a depth jump should have one signed direction."""
    values=np.asarray(field,np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Expected finite inverse depth on the shared image lattice')
    if not inventory:
        return []
    low,high=np.percentile(values,(2,98))
    scale=high-low
    if scale==0:
        scale=np.ptp(values)
    z=(values-low)/scale if scale>0 else np.zeros_like(values)
    near,far=SETTINGS['near_distance'],SETTINGS['far_distance']
    ratio=(2*near+1)/(2*(far-near))
    decisions=[]
    for p in inventory:
        pts=np.asarray(p['samples'])
        a,b,c,d=z[pts[:,::2],pts[:,1::2]].T
        jump=b-c
        step=jump-ratio*((a-b)+(c-d))
        median=float(np.median(step))
        direction=np.sign(median)
        agreement=float(np.mean((direction*step>0)&(direction*jump>0)))
        decisions.append(dict(parent=p['parent'],proposal=p['proposal'],accepted=False,children=[],
                              median_jump=float(np.median(jump)),median_step=median,sign_agreement=agreement,
                              sample_count=len(pts)))
    return decisions


def partition(image,part,proposal,mask):
    # Reuse the same native-mask partition and chart fitting as the two-shape pass.
    size=part['support']['size']
    minimum=SETTINGS['min_seed_samples']*size[0]*size[1]/mask.size
    parts=children(image,part,[0,1],np.stack((~mask,mask)),[{'quality':0.},{'quality':1.}],
                   minimum_seed_area=minimum)
    for i,child in enumerate(parts):
        seed=child.pop('mobile_sam_proposal')
        child.update(id=f"{part['id']}-single-{proposal}-{i}",single_wall_proposal=proposal,
                     single_wall_side={0:'remainder',1:'inside',None:'unresolved'}[seed])
    return parts


def apply(image,regions,masks,records,inventory,field,*,min_step=None,min_agreement=None):
    """Choose at most one accepted outline per parent; retain each connected child and hole."""
    if field.shape!=(image.height//STRIDE,image.width//STRIDE):
        raise ValueError('Depth map and candidate lattice differ')
    threshold=SETTINGS['min_step'] if min_step is None else min_step
    agreement=SETTINGS['min_agreement'] if min_agreement is None else min_agreement
    decisions=votes(inventory,field)
    best={}
    for event in decisions:
        strength=abs(event['median_step'])
        if strength<threshold or event['sign_agreement']<agreement:
            continue
        rank=(strength*event['sign_agreement'],records[event['proposal']]['quality'],-event['proposal'])
        parent=event['parent']
        if parent not in best or rank>best[parent][0]:
            best[parent]=(rank,event)
    output=[]
    for part in regions:
        if part['id'] not in best:
            output.append(part)
            continue
        event=best[part['id']][1]
        i=event['proposal']
        pieces=partition(image,part,i,masks[i])
        if not {'inside','remainder'}<={p['single_wall_side'] for p in pieces}:
            output.append(part)
            continue
        event.update(accepted=True,children=[p['id'] for p in pieces])
        output.extend(pieces)
    return output,decisions
