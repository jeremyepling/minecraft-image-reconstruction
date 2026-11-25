"""Calibrate one-block translations from textured wall immediately around each rim."""
import numpy as np
from engine.geometry import bilinear

RATIOS=np.linspace(.86,1.14,57)


def texture_detail(rect):
    gray=rect.astype(np.float32).mean(axis=2)
    return (gray-sum(np.roll(gray,k,axis=a) for k in (-2,2) for a in (0,1))/4)[:,:,None]


def local_spacing(field,front,lo,hi,pitch):
    """Use pairs with both samples on the wall, outside all candidate interiors."""
    h,w=field.shape[:2]
    low=np.maximum(2,lo-2.5*pitch)
    high=np.minimum([w-2,h-2],hi+2.5*pitch)
    x,y=np.meshgrid(np.arange(low[0],high[0],2.5),np.arange(low[1],high[1],2.5))
    points=np.column_stack((x.ravel(),y.ravel()))
    def allowed(p):
        x,y=np.floor(p).astype(int).T
        return (x>=0)&(x<w)&(y>=0)&(y<h)&front[y.clip(0,h-1),x.clip(0,w-1)]
    measurements=[]
    for axis in np.eye(2):
        valid=allowed(points)&allowed(points+axis*pitch*RATIOS.min())&allowed(points+axis*pitch*RATIOS.max())
        p=points[valid]
        if len(p)<50:
            measurements.append(None)
            continue
        signal=bilinear(field,p).ravel()
        signal-=signal.mean()
        shifts=RATIOS[:,None,None]*axis*pitch
        target=bilinear(field,p+shifts)[:,:,0]
        target-=target.mean(axis=1,keepdims=True)
        score=(target@signal)/np.maximum(1.,np.linalg.norm(target,axis=1)*np.linalg.norm(signal))
        j=int(np.argmax(score))
        if j in (0,len(score)-1) or score[j]<.35 or score[j]-np.median(score)<.12:
            measurements.append(None)
        else:
            measurements.append(dict(ratio=float(RATIOS[j]),correlation=round(float(score[j]),3),pairs=len(p)))
    return measurements
