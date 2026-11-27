"""Project first-hit block faces and fit vanilla atlas colors on alternating image patches."""
import numpy as np

from engine.surface_model import intrinsic
from engine.thin_atlas_faces import _corr,_mips
from engine.visible_horizontal_faces import _surface_masks
from engine.visible_placement import _sample_face
from engine.visible_raster import SAMPLE_STRIDE

STEP=2


class Scene:
    def __init__(self,image,parts,camera,cells):
        self.cells=sorted(cells)
        self.rgb=np.asarray(image)[::STEP,::STEP].copy()
        h,w=self.rgb.shape[:2]
        self.shape=(h,w)
        self.P=np.array(camera['P'],dtype=float,copy=True)
        self.P/=np.linalg.norm(self.P[2,:3])
        self.inv=np.linalg.inv(self.P[:,:3])
        self.eye=-self.inv@self.P[:,3]
        self.sample=self.P.copy()
        self.sample[:2]=(self.P[:2]+(.5*STEP-.5)*self.P[2])*(SAMPLE_STRIDE/STEP)
        focal=np.linalg.norm(self.P[0,:3]-image.width*.5*self.P[2,:3])
        rotation=np.linalg.solve(intrinsic(image.size,focal),self.P[:,:3])
        union,masks=_surface_masks(parts,image.size,np.arange(w)*STEP+.5,np.arange(h)*STEP+.5,rotation)
        self.union=union.ravel()
        self.masks=masks.reshape(3,-1)
        n=h*w
        self.dep=np.zeros(n)
        self.owner=np.full(n,-1,int)
        self.axes=np.full(n,-1,np.int8)
        self.back=np.full(n,-1,int)
        self.back_axis=np.full(n,-1,np.int8)
        self.back_depth=np.zeros(n)
        self.cache={}
        for number,c in enumerate(self.cells):
            idx,axis,depth=self.faces(c)
            nearest=depth>self.dep[idx]+1.e-10
            i=idx[nearest]
            self.back[i]=self.owner[i]
            self.back_axis[i]=self.axes[i]
            self.back_depth[i]=self.dep[i]
            self.owner[i],self.axes[i],self.dep[i]=number,axis[nearest],depth[nearest]
            behind=(~nearest)&(depth>self.back_depth[idx])
            i=idx[behind]
            self.back[i],self.back_axis[i],self.back_depth[i]=number,axis[behind],depth[behind]
        self.owned={}
        i=np.flatnonzero(self.owner>=0)
        i=i[np.argsort(self.owner[i],kind='stable')]
        for group in np.split(i,np.flatnonzero(np.diff(self.owner[i]))+1):
            if len(group):
                self.owned[int(self.owner[group[0]])]=group

    def faces(self,cell):
        if cell in self.cache:
            return self.cache[cell]
        idx,axes,depths=[],[],[]
        for axis in range(3):
            if cell[axis]<=self.eye[axis]<=cell[axis]+1:
                continue
            hit=_sample_face(self.sample,cell,axis,1 if self.eye[axis]>cell[axis]+1 else -1,self.shape)
            if hit is None:
                continue
            box,mask,dep,*_=hit
            y,x=np.nonzero(mask)
            idx.extend((y+box[0].start)*self.shape[1]+x+box[1].start)
            axes.extend([axis]*len(y))
            depths.extend(dep[mask])
        ids=np.asarray(idx,int)
        ii,first=np.unique(ids,return_index=True)
        answer=ii,np.asarray(axes,np.int8)[first],np.asarray(depths,float)[first]
        self.cache[cell]=answer
        return answer

    def loss(self,idx,owner,axis):
        fg=owner>=0
        inside=self.union[idx]
        orient=self.masks[:,idx].any(axis=0)
        return (fg!=inside).astype(float)+.65*(fg&inside&orient&~self.masks[axis,idx])

    def move(self,n,to):
        new,axes,depth=self.faces(to)
        idx=np.union1d(self.owned[n],new)
        owner=self.owner[idx].copy()
        axis=self.axes[idx].copy()
        z=self.dep[idx].copy()
        removed=owner==n
        previous=idx[removed]
        owner[removed]=self.back[previous]
        axis[removed]=self.back_axis[previous]
        z[removed]=self.back_depth[previous]
        offered=np.searchsorted(idx,new)
        wins=depth>z[offered]+1.e-10
        t=offered[wins]
        owner[t],axis[t]=n,axes[wins]
        changed=(owner!=self.owner[idx])|(axis!=self.axes[idx])|((owner==n)&(self.owner[idx]==n))
        return idx[changed],owner[changed],axis[changed]

    def uv(self,idx,c,axis):
        y,x=np.divmod(idx,self.shape[1])
        rays=np.column_stack((x*STEP+.5,y*STEP+.5,np.ones(len(x))))@self.inv.T
        free=[d for d in range(3) if d!=axis]
        level=c[axis]+int(self.eye[axis]>c[axis]+1)
        uv=self.eye[free]+(level-self.eye[axis])*rays[:,free]/rays[:,[axis]]
        uv=np.floor(uv*16).astype(int)%16
        return uv[:,0]+16*uv[:,1]


class FaceRenderer:
    """Fit each visible face on alternating source patches, evaluate the interleaved patches."""
    def __init__(self,scene,atlas):
        self.scene=scene
        self.atlas=atlas
        self.maps=_mips(self.atlas)
        h,w=scene.shape
        y,x=np.indices((h,w))
        self.test=((x//3+y//3)%2==1).ravel()
        self.source=scene.rgb.reshape(-1,3).astype(np.float32)
        self.image=np.full_like(self.source,40)
        self.fitted=np.zeros(len(self.source),bool)
        self.face_models={}
        for n,idx in scene.owned.items():
            for a in np.unique(scene.axes[idx]):
                sample=idx[scene.axes[idx]==a]
                mdl=self.fit(sample,scene.cells[n],a)
                self.face_models[(n,a)]=mdl
                self.image[sample],self.fitted[sample]=self.draw(sample,scene.cells[n],a,mdl)

    def fit(self,idx,cell,axis):
        sample=idx[~self.test[idx]]
        if not len(sample):
            return None
        sample=sample[np.linspace(0,len(sample)-1,min(384,len(sample)),dtype=int)]
        rgb=self.source[sample]
        mean=rgb.mean(0)
        flat=(mean,None,0.,0.)
        uv=self.scene.uv(sample,cell,axis)
        unique,group,count=np.unique(uv,return_inverse=True,return_counts=True)
        if len(unique)<12:
            return flat
        observed=(np.bincount(group,weights=rgb.mean(1))/count).astype(np.float32)
        if observed.std()<.7:
            return flat
        means=self.atlas.means
        gain=np.clip((means@mean)/np.maximum(1,(means**2).sum(1)),.30,1.15)
        cost=abs(gain[:,None]*means-mean).mean(1)+2*abs(np.log(gain/.8))
        short=np.argpartition(cost,12)[:12]
        templates=self.maps[short]
        score=_corr(templates[...,unique],observed)-cost[short,None,None]/70.
        mat,mip,rot=np.unravel_index(score.argmax(),score.shape)
        template=templates[mat,mip,rot]
        pred=template[uv]
        scale=np.clip(np.mean((pred-pred.mean())*(rgb.mean(1)-mean.mean()))/max(1.,pred.var()),0,1.3)
        return mean,template,float(pred.mean()),float(scale)

    def draw(self,idx,cell,a,model):
        if model is None:
            return np.broadcast_to(np.array([110.,110.,110.]),(len(idx),3)),False
        mean,template,offset,scale=model
        if template is None:
            return np.broadcast_to(mean,(len(idx),3)),False
        return np.clip(mean+(scale*(template[self.scene.uv(idx,cell,a)]-offset))[:,None],0,255),True

    def residual_gain(self,idx,owner,axis,n,to):
        # Only compare foreground source samples on both renders. Silhouette is scored separately.
        s=self.scene
        kept=(owner>=0)&(s.owner[idx]>=0)&s.union[idx]&self.test[idx]
        if kept.sum()<8:
            return 0.,0
        updated=np.full((len(idx),3),110.)
        for a in np.unique(axis[owner==n]):
            ix=(axis==a)&(owner==n)
            model=self.fit(idx[ix],to,a)
            updated[ix],_=self.draw(idx[ix],to,a,model)
        for num in np.unique(owner[(owner>=0)&(owner!=n)]):
            for a in np.unique(axis[owner==num]):
                ix=(owner==num)&(axis==a)
                updated[ix],_=self.draw(idx[ix],s.cells[num],a,self.face_models.get((num,a)))
        src=self.source[idx[kept]]
        old=np.abs(src-self.image[idx[kept]]).mean(1)
        new=np.abs(src-updated[kept]).mean(1)
        return round(float((old-new).mean()),3),len(src)


