"""Fit a perspective lattice from repeated image seams. No atlas or block IDs are used."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


PPU=16
UNIT=np.array([[0,0],[1,0],[1,1],[0,1]],float)


def transform(quad):
    rows,values=[],[]
    for (u,v),(x,y) in zip(UNIT,quad):
        rows.extend([[u,v,1,0,0,0,-x*u,-x*v],[0,0,0,u,v,1,-y*u,-y*v]])
        values.extend([x,y])
    return np.r_[np.linalg.solve(rows,values),1].reshape(3,3)


def project(h, points):
    p=np.asarray(points)
    v=p@h[:,:2].T+h[:,2]
    return v[...,:2]/v[...,2,None]


def bilinear(rgb,xy):
    x=np.clip(xy[...,0]-.5,0,rgb.shape[1]-1.001)
    y=np.clip(xy[...,1]-.5,0,rgb.shape[0]-1.001)
    i=x.astype(int); j=y.astype(int)
    a=(x-i)[...,None]; b=(y-j)[...,None]
    return (1-b)*((1-a)*rgb[j,i]+a*rgb[j,i+1])+b*((1-a)*rgb[j+1,i]+a*rgb[j+1,i+1])


def vp(segments):
    # Fit shared vanishing direction to architectural lines; longer edges
    # constrain orientation more accurately than one blurry face edge.
    pairs=np.asarray(segments,float)
    lengths=np.linalg.norm(pairs[:,1]-pairs[:,0],axis=1)
    points=np.pad(pairs,((0,0),(0,0),(0,1)),constant_values=1)
    lines=np.cross(points[:,0],points[:,1])/lengths[:,None]
    return np.linalg.lstsq(lines[:,:2]*lengths[:,None],-lines[:,2]*lengths,rcond=None)[0]


def snap_line(image,segment):
    """Refine a rough long-edge segment in a small normal corridor using signed contrast."""
    a,b=np.asarray(segment,float)
    delta=b-a
    direction=delta/np.linalg.norm(delta)
    normal=np.array([-direction[1],direction[0]])
    t=np.linspace(.05,.95,80)
    line=a+t[:,None]*delta
    shifts=np.arange(-3,3.1,.5)
    start,end=np.meshgrid(shifts,shifts)
    s=start.ravel(); e=end.ravel()
    mid=line[None,:,:]+((1-t)*s[:,None]+t*e[:,None])[:,:,None]*normal
    difference=bilinear(image,mid+1.4*normal)-bilinear(image,mid-1.4*normal)
    # A true long edge has consistent polarity even across local flecks.
    response=np.linalg.norm(np.clip(difference,-65,65).mean(1),axis=1)
    n=int(np.argmax(response-.20*(np.abs(s)+np.abs(e))))
    return [a+s[n]*normal,b+e[n]*normal]


def local_frame(anchor,size,vps):
    origin=np.asarray(anchor)
    columns=[]
    for axis,(end,step) in enumerate(zip(vps,size)):
        k=step/(end[axis]-origin[axis]-step)
        columns.append([k*end[0],k*end[1],k])
    return np.array([*columns,[*origin,1.]]).T


def face(image,quad,size=32):
    t=(np.arange(size)+.5)/size
    uv=np.stack(np.meshgrid(t,t),axis=-1)
    return np.uint8(np.clip(bilinear(image,project(transform(quad),uv)),0,255).round())


def box_mean(im,axis,radius):
    pad=[(0,0)]*im.ndim
    pad[axis]=(radius,radius)
    ext=np.moveaxis(np.pad(im,pad,mode='edge'),axis,0)
    integral=np.concatenate([np.zeros_like(ext[:1]),ext.cumsum(0)],axis=0)
    n=2*radius+1
    return np.moveaxis((integral[n:]-integral[:-n])/n,0,axis)


def seam_profile(rgb,mask,axis):
    # Work along the tested axis. Pool short collinear runs in the orthogonal
    # direction before differentiating, suppressing isolated texture speckles.
    rgb=np.moveaxis(rgb,axis,1)
    mask=np.moveaxis(mask,axis,1)
    smooth=box_mean(rgb,0,3)
    edge=np.zeros(mask.shape)
    for lag,weight in ((1,.55),(3,.30),(5,.15)):
        diff=np.sqrt(np.mean((smooth[:,2*lag:]-smooth[:,:-2*lag])**2,axis=2))
        edge[:,lag:-lag]+=weight*diff
    # Local normalization prevents a black window edge from dominating a
    # long series of lower-contrast joints on the same plane.
    signal=np.clip(edge/(2+box_mean(edge,1,12)),0,3)
    valid=mask.sum(0)
    avg=(mask*signal).sum(0)/np.maximum(valid,1)
    return avg*np.minimum(1,valid/max(6,valid.max()*.18))


def period_scores(profile,low,steps):
    coord=low+(np.arange(len(profile))+.5)/PPU
    fits=[]
    for step in steps:
        offsets=np.arange(80)*step/80
        k=np.arange(np.floor(coord.min()/step)-1,np.ceil(coord.max()/step)+1)
        ticks=k[None,:]*step+offsets[:,None]
        valid=(ticks>coord[0]+.15)&(ticks<coord[-1]-.15)
        center=np.interp(ticks,coord,profile)
        off=(np.interp(ticks-step*.24,coord,profile)+np.interp(ticks+step*.24,coord,profile))/2
        response=(center-off)*valid
        scores=response.sum(1)/np.maximum(valid.sum(1),1)
        idx=int(scores.argmax())
        fits.append((scores[idx],step,offsets[idx]))
    return np.array(fits)


def fit_plane(image,spec,h,metric=False):
    outer=spec['outline']
    inv=np.linalg.inv(h)
    bounds=project(inv,outer)
    lo=np.floor(bounds.min(0)*PPU)/PPU
    hi=np.ceil(bounds.max(0)*PPU)/PPU
    xs,ys=[np.arange(a,b,1/PPU)+.5/PPU for a,b in zip(lo,hi)]
    uv=np.stack(np.meshgrid(xs,ys),axis=-1)
    rect=bilinear(image,project(h,uv))
    mask=Image.new('L',(len(xs),len(ys)))
    draw=ImageDraw.Draw(mask)
    def polygon(points,value):
        xy=(project(inv,points)-lo)*PPU-.5
        draw.polygon([tuple(p) for p in xy],fill=value)
    polygon(outer,255)
    for hole in spec.get('holes',[]): polygon(hole,0)
    valid=np.asarray(mask)>0
    # Aperture perimeters supply real grid-aligned edges. Their interiors are
    # excluded, so a recessed frame cannot set the phase of the front wall.
    xr,yr=spec.get('step_ranges',[[.77,1.32],[.77,1.32]])
    steps=np.linspace(.88,1.18,151)
    profiles=[period_scores(seam_profile(rect,valid,a),start,steps if metric else np.linspace(*rng,151))
              for a,start,rng in ((1,lo[0],xr),(0,lo[1],yr))]
    if metric:
        # Parallel planes seen by one camera differ in depth. In a common
        # rectification that changes block scale equally along both axes.
        joint=profiles[0][:,0]+profiles[1][:,0]-.24*np.abs(np.log(steps))
        fits=[p[joint.argmax()].tolist() for p in profiles]
    else:
        fits=[p[p[:,0].argmax()].tolist() for p in profiles]
    step=np.array([f[1] for f in fits]); phase=np.array([f[2] for f in fits])
    grid=h@np.array([[step[0],0,phase[0]],[0,step[1],phase[1]],[0,0,1]])
    cells=[]
    for j in range(int(np.floor((lo[1]-phase[1])/step[1])),int(np.ceil((hi[1]-phase[1])/step[1]))):
        for i in range(int(np.floor((lo[0]-phase[0])/step[0])),int(np.ceil((hi[0]-phase[0])/step[0]))):
            start=(np.array([i,j])*step+phase-lo)*PPU
            stop=start+step*PPU
            left,top=np.ceil(start-.5).astype(int)
            right,bottom=np.ceil(stop-.5).astype(int)
            support=float(valid[max(0,top):max(0,bottom),max(0,left):max(0,right)].sum()/
                          max(1,(right-left)*(bottom-top)))
            if support<.10: continue
            cells.append(dict(ij=[i,j],quad=project(grid,UNIT+[i,j]).round(3).tolist(),
                              visible_fraction=round(min(1,support),3)))
    return dict(name=spec['name'],grid_to_image=grid.tolist(),axes=fits,cells=cells)


def detect(image,layout):
    vanishing=[vp([snap_line(image,seg) for seg in s]) for s in layout['directions']]
    planes=[]
    metric=None
    for spec in layout['planes']:
        if metric is None:
            h=local_frame(spec['anchor'],spec['approx_block_px'],vanishing)
            plane=fit_plane(image,spec,h)
            metric=np.array(plane['grid_to_image'])
        else:
            u,v=project(np.linalg.inv(metric),spec['anchor'])
            h=metric@np.array([[1,0,u],[0,1,v],[0,0,1]])
            plane=fit_plane(image,spec,h,metric=True)
        planes.append(plane)
    return planes


def export_geometry(image_path,layout_path,out):
    image=np.asarray(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'))
    layout=json.loads(layout_path.read_text())
    fingerprint=hashlib.sha256(image_path.read_bytes()).hexdigest()
    if layout.get('image_sha256',fingerprint)!=fingerprint:
        raise ValueError('Plane layout belongs to another image')
    planes=detect(image,layout)
    out.mkdir(parents=True,exist_ok=True)
    # Save fixed crops once. The material stage can score them but cannot move them.
    full=[cell for p in planes for cell in p['cells'] if cell['visible_fraction']>=.90]
    cols=32
    sheet=Image.new('RGB',(16*cols,16*max(1,int(np.ceil(len(full)/cols)))))
    for n,cell in enumerate(full):
        x,y=16*(n%cols),16*(n//cols)
        sheet.paste(Image.fromarray(face(image,cell['quad'],16)),(x,y))
        cell['tile_rect']=[x,y,x+16,y+16]
    sheet.save(out/'faces.png')
    info=dict(image_sha256=fingerprint,crops='faces.png',crops_sha256=hashlib.sha256((out/'faces.png').read_bytes()).hexdigest(),
              full_visibility_cutoff=.90,planes=planes)
    (out/'geometry.json').write_text(json.dumps(info,indent=1))
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['image','layout','out']: parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    export_geometry(args.image,args.layout,args.out)
