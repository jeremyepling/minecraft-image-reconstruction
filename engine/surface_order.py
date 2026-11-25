"""Order neighboring visible patches by sampled depth; distances are in native image pixels."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw


SETTINGS = dict(stride_px=4,max_intervening_px=12,min_samples=4,min_agreement=.7)


def order(planes,size,depth=None):
    """Compare opposing samples across observed patches along scanlines in both directions."""
    stride=SETTINGS['stride_px']
    w,h=((n+stride-1)//stride for n in size)
    ids=Image.new('I',(w,h))
    pen=ImageDraw.Draw(ids)
    # The smaller footprint takes precedence where two projected charts overlap.
    def area(idx):
        x,y=np.array(planes[idx]['quad']).T
        return abs(x@np.roll(y,1)-y@np.roll(x,1))
    for i in sorted(range(len(planes)),key=area,reverse=True):
        pen.polygon([tuple(pt) for pt in np.array(planes[i]['quad'])/stride],fill=i+1)
    ids=np.asarray(ids)
    field=None if depth is None else np.array(Image.fromarray(depth).resize((w,h),Image.Resampling.BILINEAR))
    samples=defaultdict(list)
    for axis in (0,1):
        slices=ids if axis==0 else ids.T
        for v,row in enumerate(slices):
            breaks=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
            last=None
            for a,b in zip(breaks[:-1],breaks[1:]):
                plane=int(row[a])-1
                if plane<0:
                    continue
                if last is not None:
                    other,end,begin=last
                    if other!=plane and (a-end)*stride<=SETTINGS['max_intervening_px']:
                        u0,u1=max(begin,end-3),min(b-1,a+2)
                        p0,p1=((v,u0),(v,u1)) if axis==0 else ((u0,v),(u1,v))
                        pair=tuple(sorted((other,plane)))
                        samples[pair].append((p0,p1) if other==pair[0] else (p1,p0))
                last=plane,b,a
    result=[]
    for (i,j),coordinates in sorted(samples.items()):
        if len(coordinates)<SETTINGS['min_samples']:
            continue
        p,q=planes[i],planes[j]
        if field is None:
            if ('corners_camera' not in p or 'corners_camera' not in q or
                    (p['component']!=q['component'] and (p['units']!='blocks' or q['units']!='blocks'))):
                continue
            # Compare perspective-correct in-plane depth at the same sample locations.
            from engine.geometry import transform,project
            def inverse_z(part,indices):
                xy=(np.asarray(indices)[:,::-1]+.5)*stride
                uv=project(np.linalg.inv(transform(part['quad'])),xy)
                z=np.asarray(part['corners_camera'])[:,2]
                return 1./(z[0]+uv[:,0]*(z[1]-z[0])+uv[:,1]*(z[3]-z[0]))
            a,b=zip(*coordinates)
            delta=inverse_z(p,a)-inverse_z(q,b)
        else:
            delta=np.array([field[a]-field[b] for a,b in coordinates])
        sign=np.sign(np.median(delta))
        agreement=float(np.mean(sign*delta>0))
        if agreement>=SETTINGS['min_agreement']:
            near,far=(p,q) if sign>0 else (q,p)
            result.append(dict(near=near['id'],far=far['id'],samples=len(delta),agreement=agreement))
    return result
