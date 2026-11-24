"""Propose visible vertical walls from a silhouette and measure block spacing per wall."""
import numpy as np
from PIL import Image, ImageFilter

from engine.geometry import bilinear, project, seam_profile, transform


def silhouette(image):
    """Compare each scanline with its margins to discard sky and level terrain."""
    w, h = image.size
    k = min(1., 320/max(w,h))
    small = image.resize((max(1,round(w*k)),max(1,round(h*k))), Image.Resampling.BOX)
    rgb = np.asarray(small.convert('RGB'),dtype=float)
    sh, sw = rgb.shape[:2]
    full = [[0,0],[w,0],[w,h],[0,h]]
    if min(sh,sw)<16:
        return [full],None
    band = max(2, sw//30)
    edges = np.concatenate([rgb[:, :band],rgb[:, -band:]],axis=1)
    bg = np.median(edges,axis=1)
    diff = abs(rgb-bg[:,None]).mean(2)
    noise = np.median(abs(edges-bg[:,None]).mean(2),axis=1)
    mask = diff > np.maximum(17,noise[:,None]*2.8)
    alpha = np.asarray(small.convert('RGBA'))[:,:,3]
    if alpha.min()<32:
        mask = alpha>128
    else:
        # Smooth sky, moving clouds and phone controls often disagree with
        # both margins. Building textures usually retain local edge energy.
        luminance=rgb.mean(2)
        dx,dy=np.gradient(luminance)
        energy=Image.fromarray(np.uint8(np.minimum(255,8*np.hypot(dx,dy))))
        detailed=np.asarray(energy.filter(ImageFilter.BoxBlur(2)))>30
        if detailed.mean()>.78:
            return [full],None
        if np.median(noise)>13 or mask.mean()<.06:
            # Measure fine texture before downsampling. Clouds can have strong
            # large-scale gradients while containing very little fine detail.
            source=image.convert('L')
            residual=np.abs(np.asarray(source,dtype='i2')-
                            np.asarray(source.filter(ImageFilter.GaussianBlur(1.7)),dtype='i2'))
            texture=Image.fromarray(np.uint8(np.minimum(255,8*residual))).resize(
                small.size,Image.Resampling.BOX).filter(ImageFilter.BoxBlur(3))
            mask=np.asarray(texture)>25
        else:
            # The central object may include areas of uniform concrete.
            mask=(mask & (diff>np.maximum(25,noise[:,None]*3))) | detailed
        mask=np.asarray(Image.fromarray(mask).convert('L').filter(ImageFilter.MaxFilter(5))
                        .filter(ImageFilter.MinFilter(5)))>0
    # Largest connected foreground after a small majority filter.
    padded=np.pad(mask,1)
    mask=sum(padded[dy:dy+sh,dx:dx+sw].astype(int) for dx in range(3) for dy in range(3))>=5
    best=[]
    for y,x in zip(*np.where(mask)):
        if not mask[y,x]:
            continue
        stack=[(y,x)]; mask[y,x]=False; cells=[]
        while stack:
            yy,xx=stack.pop(); cells.append((yy,xx))
            for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0<=ny<sh and 0<=nx<sw and mask[ny,nx]:
                    stack.append((ny,nx)); mask[ny,nx]=False
        if len(cells)>len(best): best=cells
    if len(best)<.06*sw*sh: return [full],None
    yy,xx=np.array(best).T
    foreground=np.zeros((sh,sw),dtype=np.uint8)
    foreground[yy,xx]=255
    foreground=np.asarray(Image.fromarray(foreground).resize((w,h),Image.Resampling.NEAREST))
    xs=np.arange(xx.min(),xx.max()+1)
    top=np.array([yy[xx==x].min() for x in xs]); bottom=np.array([yy[xx==x].max()+1 for x in xs])
    if len(xs)<8: return [full],None
    # A roofline corner can divide a returning wall from the front elevation.
    def line(x,y):
        coef=np.polyfit(x,y,1)
        residual=np.abs(y-np.polyval(coef,x))
        keep=residual<=max(2,np.percentile(residual,75))
        return np.polyfit(x[keep],y[keep],1) if keep.sum()>2 else coef
    def fit(a,b):
        x=xs[a:b]
        upper=line(x,top[a:b]); lower=line(x,bottom[a:b])
        err=np.minimum(20,abs(top[a:b]-np.polyval(upper,x))).sum()
        return upper,lower,err
    n=len(xs); whole=fit(0,n)
    segments=[(0,n)]
    for j in sorted(range(max(4,n//6),n-max(4,n//6),2),key=lambda j:fit(0,j)[2]+fit(j,n)[2]):
        l,r=fit(0,j),fit(j,n)
        if (l[2]+r[2] < .55*whole[2] and l[0][0]<-.12 and r[0][0]>.08
                and abs(np.polyval(l[0]-r[0],xs[j])) < sh*.16):
            segments=[(0,j),(j,n)]
        break
    result=[]
    for a,b in segments:
        upper,lower,_=fit(a,b)
        x0,x1=xs[a],xs[b-1]+1
        q=np.array([[x0,np.polyval(upper,x0)],[x1,np.polyval(upper,x1)],
                    [x1,np.polyval(lower,x1)],[x0,np.polyval(lower,x0)]])
        if len(segments)==2:
            junction=segments[1][0]
            top_corner=np.median(top[junction-2:junction+3])
            base_corner=np.median(bottom[junction-2:junction+3])
            if a==junction: q[[0,3],1]=[top_corner,base_corner]
            else: q[[1,2],1]=[top_corner,base_corner]
        if min(q[2:,1])-max(q[:2,1])<3: return [full],None
        result.append((np.clip(q,[0,0],[sw,sh])*[w/sw,h/sh]).round(2).tolist())
    return result,foreground


def estimate_grid(rgb, quad):
    q=np.array(quad,float)
    sides=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
    w,h=np.maximum(4,np.rint([sides[[1,3]].mean(),sides[[0,2]].mean()])).astype(int)
    factor=min(1,640/max(w,h)); w,h=max(4,round(w*factor)),max(4,round(h*factor))
    xs,ys=np.meshgrid((np.arange(w)+.5)/w,(np.arange(h)+.5)/h)
    rect=bilinear(rgb,project(transform(q),np.stack((xs,ys),axis=-1)))
    result=[]
    alternatives=[]
    for axis in (1,0):
        profile=seam_profile(rect,np.ones((h,w),bool),axis)
        length=len(profile); scores=[]
        # Very wide bays and storey spacing are architectural repeats too.
        # Require finer repeats on large wall crops before accepting a grid.
        for n in range(max(2,int(np.ceil(length/(80*factor)))),min(128,length//5)+1):
            step=length/n; ticks=np.arange(1,n)*step-.5
            center=np.interp(ticks,np.arange(length),profile)
            away=np.mean([np.interp(ticks+step*t,np.arange(length),profile)
                          for t in (-.33,-.2,.2,.33)],axis=0)
            scores.append((float((center-away).mean()*min(1,(n-1)/4)),n))
        score,n=max(scores,default=(0.,1))
        if score<.08:
            n=max(1,round(length/(16*factor)))
        result.append((n,round(score,3)))
        alternatives.append(scores)
    # Vertical walls viewed from the side foreshorten their horizontal axis.
    # A much wider-than-tall cell is often an aperture spanning several blocks.
    horizontal,vertical=result
    ratio=(w/h)*(vertical[0]/horizontal[0])
    if ratio>1.65 and vertical[1]>=.08:
        compatible=[(score,n) for score,n in alternatives[0]
                    if .35<=(w/h)*(vertical[0]/n)<=1.5]
        if compatible:
            score,n=max(compatible)
            if score>=.08: result[0]=(n,round(score,3))
    return result
