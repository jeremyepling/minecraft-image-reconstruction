"""Back-project a photographed roof skyline onto competing anchored horizontal courses."""
from collections import defaultdict

import numpy as np

from engine.sky_filter import sky_masks
from engine.surface_model import intrinsic
from engine.visible_horizontal_faces import _keep_prior_visible, _surface_masks
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


class Skyline:
    def __init__(self,image,parts,camera,prior):
        self.size=image.size
        self.shape=image.size[::-1]
        self.prior=prior
        self.P=np.array(camera['P'],float)
        self.P/=np.linalg.norm(self.P[2,:3])
        self.inv=np.linalg.inv(self.P[:,:3])
        self.eye=-self.inv@self.P[:,3]
        self.native=self.P.copy()
        self.native[:2]*=SAMPLE_STRIDE
        self.rgb=np.asarray(image.convert('RGB'),float)
        blue,cloud=sky_masks(image)
        self.sky=blue|cloud
        self.skyline=np.argmax(~self.sky,axis=0)
        self.valid=self.sky[0] & (self.skyline>2) & (self.skyline<self.shape[0]-2)
        focal=np.linalg.norm(self.P[0,:3]-image.width*.5*self.P[2,:3])
        rotation=np.linalg.solve(intrinsic(self.size,focal),self.P[:,:3])
        self.union,self.masks=_surface_masks(parts,self.size,np.arange(image.width)+.5,
                                             np.arange(image.height)+.5,rotation)
        self.points=sorted(prior)
        self.owner,self.axes=_frontmost_cells(self.points,self.native,self.eye,self.shape,face_axes=True)
        self.points=np.asarray(self.points)
        self.levels=defaultdict(set)
        yy,xx=np.nonzero((self.owner>=0)&(self.axes==1))
        for i in np.unique(self.owner[yy,xx]):
            x,y,z=map(int,self.points[i])
            if y>0 and y+1<self.eye[1]:
                self.levels[y].add((x,z))
        self.depth=np.zeros(self.shape)
        yy,xx=np.nonzero(self.owner>=0)
        a=self.axes[yy,xx]
        c=self.points[self.owner[yy,xx],a]
        self.depth[yy,xx]=self.rays(xx,yy)[np.arange(len(a)),a]/(c+(self.eye[a]>c+1)-self.eye[a])

    def rays(self,x,y):
        return np.column_stack((x+.5,y+.5,np.ones(len(x))))@self.inv.T

    def project(self,world):
        pixel=world@self.P[:,:3].T+self.P[:,3]
        return pixel[:,:2]/pixel[:,[2]]

    def rim_options(self):
        # Integer edges compete on the exact same source skyline. Roof height
        # and footprint are never supplied by the scorer.
        xs=np.flatnonzero(self.valid)
        if not len(xs):
            return []
        ys=self.skyline[xs]
        rays=self.rays(xs,ys-.5)
        result=[]
        for y,anchors in self.levels.items():
            if len(anchors)<3:
                continue
            with np.errstate(divide='ignore'):
                dist=(y+1-self.eye[1])/rays[:,1]
            ok=(dist>1)&(dist<130)
            xz=self.eye+dist[ok,None]*rays[ok]
            for axis,tangent in ((0,2),(2,0)):
                edges=np.rint(xz[:,axis])
                q=xz.copy()
                q[:,axis]=edges
                pixels=self.project(q)
                # Compare the projected line to the source edge at the same
                # screen x, accounting for its vanishing direction.
                q[:,tangent]+=1.
                v=self.project(q)-pixels
                slope=v[:,1]/np.where(abs(v[:,0])>.01,v[:,0],np.nan)
                error=abs(ys[ok]-(pixels[:,1]+(xs[ok]+.5-pixels[:,0])*slope))
                choices=defaultdict(lambda:defaultdict(list))
                for i in np.flatnonzero(error<=1.25):
                    edge=int(edges[i])
                    along=int(np.floor(xz[i,tangent]))
                    choices[edge][along].append(float(error[i]))
                for edge,bins in choices.items():
                    supported=sorted(k for k,err in bins.items() if len(err)>=3)
                    for group in np.split(supported,np.flatnonzero(np.diff(supported)!=1)+1):
                        if len(group)<2:
                            continue
                        toward=1 if self.eye[axis]>edge else -1
                        outer=edge if toward>0 else edge-1
                        row=[]
                        for length in group.tolist():
                            for d in (0,1):
                                point=[0,y,0]
                                point[axis]=outer+d*toward
                                point[tangent]=length
                                row.append(tuple(point))
                        # An already visible top at this height must adjoin this strip.
                        nearby={a for x,_,z in row for dx,dz in ((0,0),(1,0),(-1,0),(0,1),(0,-1))
                                if (a:=(x+dx,z+dz)) in anchors}
                        if nearby:
                            samples=[v for k in group for v in bins[k]]
                            result.append(dict(cells=set(row)-self.prior,anchors=len(nearby),span=len(group),
                                               residual=float(np.median(samples))))
        return result

    def face(self,c,joint=None):
        hit=_sample_face(self.native,c,1,1,self.shape)
        if hit is None:
            return None
        box,face,dep,*_=hit
        seen=face&(dep>self.depth[box]+1e-8)
        if joint is not None:
            owner,axes,idx=joint
            seen &= (owner[box]==idx[c]) & (axes[box]==1)
        yy,xx=np.nonzero(seen)
        if len(xx)<3:
            return None
        yy,xx=yy+box[0].start,xx+box[1].start
        if self.sky[yy,xx].any():
            return None
        oriented=self.masks[1,yy,xx]
        edge=(yy>=self.skyline[xx])&(yy<self.skyline[xx]+3)&self.valid[xx]
        if (oriented|edge).mean()<.82:
            return None
        return xx,yy

    def repeated(self,c,xy):
        x,y=xy
        if len(x)<6:
            return False
        sel=np.linspace(0,len(x)-1,min(128,len(x)),dtype=int)
        x,y=x[sel],y[sel]
        rays=self.rays(x,y)
        world=self.eye+(c[1]+1-self.eye[1])*rays/rays[:,[1]]
        # Ask a previously photographed neighbor for the color at the same
        # within-block coordinates. Include shading and repeat phase.
        for dx,dz in ((1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2)):
            donor=c[0]+dx,c[1],c[2]+dz
            if donor not in self.prior:
                continue
            q=np.floor(self.project(world+[dx,0,dz])).astype(int)
            u,v=q.T
            keep=(u>=0)&(u<self.size[0])&(v>=0)&(v<self.size[1])
            u,v=u[keep],v[keep]
            ok=(self.axes[v,u]==1)&(self.owner[v,u]>=0)&np.all(self.points[self.owner[v,u]]==donor,axis=1)
            if ok.sum()<6 or ok.sum()<.6*len(x):
                continue
            actual=self.rgb[y[keep][ok],x[keep][ok]]
            expected=self.rgb[v[ok],u[ok]]
            a,b=actual.mean(1),expected.mean(1)
            if min(a.std(),b.std())<1.2:
                continue
            if np.corrcoef(a,b)[0,1]>.72 and np.mean(abs(actual-expected))<7:
                return True
        return False

    def finalize(self,selected,verify_texture):
        cells=set(selected)
        while cells:
            ordered=sorted(self.prior|cells)
            owner,axes=_frontmost_cells(ordered,self.native,self.eye,self.shape,face_axes=True)
            idx={c:i for i,c in enumerate(ordered)}
            spill=set(np.unique(owner[self.sky & (self.owner<0)]))
            keep=set()
            for c in cells:
                if idx[c] in spill:
                    continue
                xy=self.face(c,(owner,axes,idx))
                if xy is not None and (not verify_texture or self.repeated(c,xy)):
                    keep.add(c)
            keep=_keep_prior_visible(self.size,self.P,self.eye,self.prior,keep)-self.prior if keep else set()
            if keep==cells:
                break
            cells=keep
        return cells


def proposals(image,parts,camera,prior):
    if not prior or camera.get('P') is None or not parts:
        return None,set(),set()
    ctx=Skyline(image,parts,camera,prior)
    pool=set()
    predicted=set()
    cache={}
    for row in ctx.rim_options():
        for c in row['cells']:
            if c not in cache:
                xy=ctx.face(c)
                cache[c]=(xy is not None,xy is not None and ctx.repeated(c,xy))
            valid,repeat=cache[c]
            if valid:
                pool.add(c)
                if repeat and row['anchors']>=2 and row['span']>=3 and row['residual']<=.75:
                    predicted.add(c)
    return ctx,pool,predicted


def recover(image,parts,counts,camera,prior):
    ctx,pool,predicted=proposals(image,parts,camera,prior)
    kept=ctx.finalize(predicted,True) if predicted else set()
    return prior|kept,dict(proposed=len(pool),matched=len(predicted),kept=len(kept))


def supported(ctx,cells,*,between):
    """Require the new top to repair a photographed gap between measured rim cells."""
    accepted=set(cells)
    while accepted:
        ordered=sorted(ctx.prior|accepted)
        owner,axes=_frontmost_cells(ordered,ctx.native,ctx.eye,ctx.shape,face_axes=True)
        idx={c:i for i,c in enumerate(ordered)}
        keep=set()
        for c in accepted:
            x,y,z=c
            anchors=ctx.levels[y]
            # These bounds measure the footprint along both image-derived grid axes.
            if not (any(a<=x for a,b in anchors) and any(a>=x for a,b in anchors)
                    and any(b<=z for a,b in anchors) and any(b>=z for a,b in anchors)):
                continue
            if between and not any(
                any((x+dx*k,z+dz*k) in anchors for k in range(1,6))
                and any((x-dx*k,z-dz*k) in anchors for k in range(1,6))
                for dx,dz in ((1,0),(0,1))):
                continue
            xx,yy=(None,None)
            face=ctx.face(c,(owner,axes,idx))
            if face is None:
                continue
            xx,yy=face
            fresh=ctx.owner[yy,xx]<0
            if fresh.sum()<3 or fresh.mean()<.8:
                continue
            vv,uu=np.nonzero(owner==idx[c])
            if np.any(ctx.sky[vv,uu]):
                continue
            sides=axes[vv,uu]!=1
            if sides.sum()>2:
                oriented=ctx.masks[axes[vv[sides],uu[sides]],vv[sides],uu[sides]]
                if oriented.mean()<.94:
                    continue
            if not between and not ctx.repeated(c,face):
                continue
            if between and ctx.masks[1,yy,xx].mean()<.95:
                continue
            keep.add(c)
        keep=ctx.finalize(keep,not between)
        if keep==accepted:
            break
        accepted=keep
    return accepted


def strict(image,parts,counts,camera,prior):
    ctx,pool,predicted=proposals(image,parts,camera,prior)
    kept=supported(ctx,predicted,between=False) if predicted else set()
    return prior|kept,dict(proposed=len(pool),matched=len(predicted),kept=len(kept))


def bracket(image,parts,counts,camera,prior):
    ctx,pool,_=proposals(image,parts,camera,prior)
    kept=supported(ctx,pool,between=True) if pool else set()
    return prior|kept,dict(proposed=len(pool),kept=len(kept))
