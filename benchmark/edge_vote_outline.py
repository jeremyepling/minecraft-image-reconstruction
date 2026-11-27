"""Vote along straight shared edges of consecutive, source-supported corner blocks."""
from collections import defaultdict
from itertools import groupby, permutations

import numpy as np
from PIL import ImageFilter

from engine.pillar_seams import native_mask, orientation
from engine.plane_consensus import row_runs


def project(points, projection):
    out = points @ projection[:, :3].T + projection[:, 3]
    return out[..., :2] / out[..., 2:]


def sample(array, xy):
    x = np.clip(np.floor(xy[...,0]).astype(int), 0, array.shape[1]-1)
    y = np.clip(np.floor(xy[...,1]).astype(int), 0, array.shape[0]-1)
    return array[y,x]


def exposed_edges(image, parts, counts, camera, cells):
    """Return straight cube edges visible from two sides, independent of scoring labels."""
    owner = np.zeros(image.size[::-1],np.int32)
    for index,part in enumerate(parts,1):
        mask = native_mask(part,image.size)
        owner[mask] = np.where(owner[mask]==0,index,-1)
    if not cells or not camera or camera.get('P') is None:
        return owner,[]
    p,rot=orientation(camera,image.size)
    eye=np.linalg.solve(p[:,:3],-p[:,3])
    axes=[-1]+[int(np.argmax(abs(rot.T @ a['normal_camera']))) if a.get('normal_camera') is not None else -1
               for a in parts]
    dimensions={r['id']:r['dimensions']['blocks'] for r in counts['surfaces']}
    measured=np.array([False]+[None not in dimensions.get(a['id'],[None,None]) for a in parts])
    coords=np.array(sorted(cells),int)
    toward=np.sign(eye-coords-.5).astype(int)
    front=np.array([[tuple(c+np.eye(3,dtype=int)[axis]*t[axis]) not in cells
                     and abs(eye[axis]-c[axis]-.5)>.5 for axis in range(3)] for c,t in zip(coords,toward)])
    groups=defaultdict(list)
    for host,side in permutations(range(3),2):
        run=3-host-side
        visible=front[:,host]&front[:,side]
        cc,tt=coords[visible],toward[visible]
        if not len(cc):
            continue
        origin=cc.copy()
        for k in (host,side):
            origin[:,k]+=tt[:,k]>0
        a=project(origin,p)
        b=project(origin+np.eye(3)[run],p)
        length=np.linalg.norm(b-a,axis=1)
        tangent=(b-a)/np.maximum(1.e-8,length[:,None])
        normal=np.stack([-tangent[:,1],tangent[:,0]],axis=-1)
        midpoint=origin+np.eye(3)[run]*.5
        center=project(midpoint,p)
        # The host extends along the side axis; the narrow perpendicular face along the host axis.
        hext=-tt[:,side,None]*np.eye(3)[side]
        sext=-tt[:,host,None]*np.eye(3)[host]
        hwidth=np.sum((project(midpoint+hext,p)-center)*normal,axis=1)
        swidth=np.sum((project(midpoint+sext,p)-center)*normal,axis=1)
        visible=(length>=7)&(abs(hwidth)>=6)&(hwidth*swidth<0)&(abs(swidth)>=1.5)&(abs(swidth)<=.75*abs(hwidth))
        def labels(extension,axis,minimum):
            offsets=np.array([np.eye(3)[run]*t for t in (.20,.50,.80)])
            samples=origin[:,None,None,:]+offsets[None,:,None,:]+extension[:,None,None,:]*np.array([.3,.68])[None,None,:,None]
            votes=sample(owner,project(samples,p)).reshape(-1,6)
            choices=np.array([np.bincount(v[v>0]).argmax() if (v>0).any() else 0 for v in votes])
            return choices,(np.sum(votes==choices[:,None],axis=1)>=minimum)&(np.array(axes)[choices]==axis)
        h,yes=labels(hext,host,5)
        s,agree=labels(sext,side,4)
        for i in np.flatnonzero(visible&yes&agree&measured[h]):
            key=(int(h[i]),int(s[i]),host,side,int(origin[i,host]),int(origin[i,side]))
            groups[key].append((int(cc[i,run]),a[i],b[i],np.sign(swidth[i])*normal[i],abs(swidth[i]),tuple(cc[i].tolist())))
    runs=[]
    for key,items in sorted(groups.items()):
        for _,group in groupby(enumerate(sorted(items,key=lambda row:row[0])),key=lambda pair:pair[1][0]-pair[0]):
            chain=[e for _,e in group]
            if len(chain)>=2:
                runs.append((key,chain))
    return owner,runs


def votes(edge,rgb):
    _,a,b,normal,width,_=edge
    length=np.linalg.norm(b-a)
    points=a+(b-a)*np.linspace(.15,.85,max(5,round(.7*length)))[:,None]
    # Adjacent pixel bands across each proposed line, sampled away from block junctions.
    offsets=np.arange(-3,4)
    center=points[:,None,:]+normal*offsets[:,None]
    left=sample(rgb,center-normal*1.25)
    right=sample(rgb,center+normal*1.25)
    profile=np.abs(right-left).mean(2)
    return profile[::2].mean(0),profile[1::2].mean(0)


def adjustments(image,parts,counts,camera,cells):
    owner,runs=exposed_edges(image,parts,counts,camera,cells)
    native_rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    rgb=np.asarray(image.convert('RGB').filter(ImageFilter.GaussianBlur(.7)),dtype=np.float32)
    proposals=[]
    for key,edges in runs:
        host,side=key[:2]
        profiles=[votes(edge,rgb) for edge in edges]
        train,held=np.moveaxis(np.array(profiles),1,0)
        offset=int(np.argmax(train.mean(0)))-3
        # Permit only a small shift from the reprojected geometric edge.
        agreed=np.abs(train.argmax(1)-3-offset)<=1
        confirmed=np.abs(held.argmax(1)-3-offset)<=1
        strength=held[:,3+offset]
        good=agreed&confirmed&(strength>=5)
        for _,group in groupby(enumerate(edges),key=lambda pair:bool(good[pair[0]])):
            group=list(group)
            if not good[group[0][0]] or len(group)<2:continue
            chain=[e for i,e in group]
            ids=[i for i,e in group]
            margin=np.min(strength[ids]-np.minimum(held[ids,0],held[ids,-1]))
            if abs(offset)>2 or margin<1:continue
            start,end=chain[0][1],chain[-1][2]
            normal=chain[0][3]
            tangent=(end-start)/np.linalg.norm(end-start)
            xmin,ymin=np.maximum(0,np.floor(np.minimum(start,end)-6).astype(int))
            xmax,ymax=np.minimum(image.size,np.ceil(np.maximum(start,end)+6).astype(int))
            y,x=np.mgrid[ymin:ymax,xmin:xmax]
            xy=np.stack([x+.5,y+.5],-1)
            distance=(xy-start)@normal-offset
            at=(xy-start)@tangent
            # A transfer is possible only if we find the two distinct observed owners on opposite sides.
            footprint=(abs(distance)<=3)&(at>=1)&(at<=np.linalg.norm(end-start)-1)
            confirm=(sample(owner,xy-(distance+3.4)[...,None]*normal)==host)&(sample(owner,xy+(3.4-distance)[...,None]*normal)==side)
            before=owner[ymin:ymax,xmin:xmax]
            after=np.where(distance>=0,side,host)
            move=footprint&confirm&((before==host)|(before==side))&(before!=after)
            if not move.any():continue
            yy,xx=y[move],x[move]
            positions=xy[move]
            along=distance[move]
            seam=positions-along[:,None]*normal
            # Validate individual pixel ownership using interior colors on both sides.
            # A two-pixel uncertainty band discards borderline antialiasing and raster rounding.
            bands=np.array([3.4,4.8,6.2])[:,None]*normal
            hband=sample(native_rgb,seam[:,None,:]-bands)
            sband=sample(native_rgb,seam[:,None,:]+bands)
            hcolor=np.median(hband,axis=1)
            scolor=np.median(sband,axis=1)
            dst=np.where((along>=0)[:,None],scolor,hcolor)
            src=np.where((along>=0)[:,None],hcolor,scolor)
            distance_new=abs(native_rgb[yy,xx]-dst).mean(1)
            distance_old=abs(native_rgb[yy,xx]-src).mean(1)
            consistent=((sample(owner,seam[:,None,:]-bands)==host)&(sample(owner,seam[:,None,:]+bands)==side)).all(1)
            clear=(abs(along)>=1.5)&consistent&(distance_old>=12)&(distance_new<=.25*distance_old)
            proposals.append(dict(host=int(host),side=int(side),cells=len(chain),offset=offset,
                host_axis=key[2],side_axis=key[3],members=[e[5] for e in chain],
                margin=round(float(margin),3),strength=round(float(min(strength[ids])),3),
                projected_width=round(float(min(e[4] for e in chain)),3),
                pixels=len(xx),xyxy=[int(xmin),int(ymin),int(xmax),int(ymax)],
                flat=yy*image.width+xx,after=after[move],clear=clear))
    return owner,runs,proposals


def revised_parts(parts,old,new):
    touched=set(old[old!=new])|set(new[old!=new])
    result=[]
    for index,part in enumerate(parts,1):
        if index not in touched:
            result.append(part)
            continue
        # Change ownership only on contested edge pixels, keep the original sampling chart.
        mask=native_mask(part,old.shape[::-1])
        mask[old!=new]=new[old!=new]==index
        clean={k:v for k,v in part.items() if k not in ('sampling_chart','source_crop','context_crop','outline')}
        clean.update(support=dict(size=list(old.shape[::-1]),runs=row_runs(mask)),support_area=int(mask.sum()))
        result.append(clean)
    return result
