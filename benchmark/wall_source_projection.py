"""Render visible exports from the estimated source camera and grade image-space depth alternatives."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import STABLE,checked,digest,read,save
from benchmark.texture_camera_sheet import font
from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.thin_atlas_faces import _corr,_mips
from engine.visible_horizontal_faces import _surface_masks
from engine.visible_placement import _sample_face
from engine.visible_raster import SAMPLE_STRIDE

RUN=Path('../facade-placement-round/runs/production')
REVIEW=Path('benchmark/reviews/wall-source-projection-01')
LABELS=('Colonnade 01','Dev Relief 00','Relief 00','Facade')
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
    def __init__(self,scene):
        self.scene=scene
        self.atlas=_atlas('26.3',STABLE/'runs/26.3/atlas')
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


def measure_moves(scene,renderer,*,minimum_geometry=None):
    """Score plus/minus one block along the most visible vertical face's normal."""
    cells=set(scene.cells)
    trials=[]
    for n,idx in scene.owned.items():
        votes=np.bincount(scene.axes[idx],minlength=3)
        a=0 if votes[0]>=votes[2] else 2
        if votes[a]<8:
            continue
        old=scene.cells[n]
        for offset in (-1,1):
            to=list(old)
            to[a]+=offset
            to=tuple(to)
            if to in cells:
                continue
            changed,owner,axis=scene.move(n,to)
            fresh=owner==n
            if fresh.sum()<8 or not (axis[fresh]!=1).any():
                continue
            match=scene.masks[axis[fresh],changed[fresh]]
            if match.mean()<.60:
                continue
            baseline=scene.loss(changed,scene.owner[changed],scene.axes[changed])
            new=scene.loss(changed,owner,axis)
            gain=float((baseline-new).sum())
            if minimum_geometry is not None and gain < minimum_geometry:
                continue
            chroma,count=renderer.residual_gain(changed,owner,axis,n,to)
            trials.append(dict(old=old,to=to,geometry_gain=round(gain,2),rgb_gain=chroma,
                               held_out=count,touched=len(changed),support=round(float(match.mean()),3)))
    return trials


def audit(job):
    split,f,prior,row,run,out,name=job
    src=checked(run/split/f['id']/'reconstruction.zip',row['pack_sha256'])
    cells=set(decoded_pack(src))
    image,parts,counts,camera=framed.photo(split,f,prior)
    assert row['rgb_sha256']==prior['rgb_sha256']
    if camera.get('P') is None:
        assert not cells
        return dict(id=f['id'],camera=False,metrics=row['relative_geometry'])
    scene=Scene(image,parts,camera,cells)
    predicted=scene.owner>=0
    mask_iou=float((predicted&scene.union).sum()/max(1,(predicted|scene.union).sum()))
    renderer=FaceRenderer(scene) if name else None
    trials=measure_moves(scene,renderer) if renderer else []
    # The recorded camera and visible labels are strictly downstream of all candidate scores.
    _,_,_,_,pose,labels,ids,truth,_,scored,_=read_view(run,split,f,prior,row)
    T=registration(scene.P,pose)
    registered=np.floor((np.asarray(scene.cells)+.5)@T[:3,:3].T+T[:3,3]).astype(int)
    truth_sample=ids[::STEP,::STEP].ravel()
    inside=truth_sample>0
    shared=inside&predicted
    faces=labels['faces']
    target_xyz=np.zeros((len(faces)+1,3),int)
    target_axis=np.full(len(faces)+1,-1,int)
    for face in faces:
        target_xyz[face['id']]=face['xyz_world']
        target_axis[face['id']]=int(np.argmax(np.abs(face['normal_world'])))
    local_to_world=np.argmax(abs(T[:3,:3]),axis=0)
    matched_axis=(local_to_world[scene.axes]==target_axis[truth_sample])&shared
    matched_cell=np.all(registered[scene.owner]==target_xyz[truth_sample],axis=1)&shared
    stats=dict(target_pixels=int(inside.sum()),placed_pixels=int(predicted.sum()),shared=int(shared.sum()),
               silhouette_iou=round(float(shared.sum()/max(1,(inside|predicted).sum())),5),
               internal_orientation_exact=int(matched_axis.sum()),first_hit_cell_exact=int(matched_cell.sum()),
               same_orientation_wrong_cell=int((matched_axis&~matched_cell).sum()),estimated_mask_iou=round(mask_iou,5))
    result=dict(id=f['id'],split=split,camera=True,pack_sha256=row['pack_sha256'],rgb_sha256=row['rgb_sha256'],
                metrics=row['relative_geometry'],projection=stats)
    if name:
        shift=np.asarray(row['translation'])
        def mapped(c):
            return tuple((np.floor((np.asarray(c)+.5)@T[:3,:3].T+T[:3,3]).astype(int)+shift).tolist())
        for t in trials:
            t['old_correct']=mapped(t['old']) in truth
            t['new_correct']=mapped(t['to']) in truth
        test=renderer.test&renderer.fitted&predicted&scene.union
        stats.update(held_out_textured_samples=int(test.sum()),held_out_rgb_mae=round(float(np.abs(
            renderer.source[test]-renderer.image[test]).mean()),3))
        result.update(name=name,trials=trials)
        panel(out,name,scene,renderer,inside,predicted,matched_axis,matched_cell,stats,row)
    return result


def panel(out,name,s,rend,truth,pred,axis,cell,stats,row):
    h,w=s.shape
    src=Image.fromarray(s.rgb)
    show=(s.union|pred|truth).reshape(h,w)
    yy,xx=np.nonzero(show)
    box=(max(0,min(xx)-8),max(0,min(yy)-8),min(w,max(xx)+9),min(h,max(yy)+9))
    lit=np.array([[146,148,158],[196,199,206],[110,112,125]],np.uint8)
    shape=s.rgb.reshape(-1,3).copy()
    shape[pred]=lit[s.axes[pred]]
    shade=rend.source.copy()
    shade[pred]=rend.image[pred]
    overlay=s.rgb.reshape(-1,3).copy()
    overlay[truth&~pred]=[65,166,254]
    overlay[pred&~truth]=[223,78,82]
    overlay[pred&truth&~axis]=[236,147,54]
    overlay[axis&~cell]=[202,90,200]
    overlay[cell&axis]=(.75*overlay[cell&axis]+.25*np.array([80,190,135])).astype(np.uint8)
    marks=s.rgb.reshape(-1,3).copy()
    marks[s.union&~pred]=[49,156,247]
    marks[pred&~s.union]=[241,86,68]
    bad=pred&s.union&s.masks.any(0)&~s.masks[s.axes,np.arange(len(pred))]
    marks[bad]=[239,179,49]
    pane=Image.new('RGB',(1250,348),'#172731')
    pen=ImageDraw.Draw(pane)
    pen.text((12,7),name,font=font(22),fill='white')
    tiles=[(src,'Source crop'),(Image.fromarray(shape.reshape(h,w,3)),'Estimate-camera geometry'),
           (Image.fromarray(shade.reshape(h,w,3).astype(np.uint8)),'Appearance fitted to source'),
           (Image.fromarray(marks.reshape(h,w,3)),'Image-only disagreement'),
           (Image.fromarray(overlay.reshape(h,w,3)),'First-hit reference errors')]
    for i,(im,title) in enumerate(tiles):
        x=10+248*i
        pen.text((x,40),title,font=font(16),fill='white')
        tile=ImageOps.contain(im.crop(box),(237,218),Image.Resampling.NEAREST)
        pane.paste(tile,(x+(237-tile.width)//2,70+(218-tile.height)//2))
    pen.text((12,297),f"Estimated mask IoU {stats['estimated_mask_iou']:.1%}     Recorded silhouette IoU {stats['silhouette_iou']:.1%}     Correct 3D block on shared rays {stats['first_hit_cell_exact']/stats['shared']:.1%}     Structure F1 {row['f1']:.1%}",font=font(17),fill='white')
    pen.text((12,325),'Blue = missing    Red = spill    Orange = differing face orientation    Purple = same orientation, wrong block (reference only)',font=font(15),fill='#c9d8df')
    pane.save(out/(name.lower().replace(' ','-')+'.jpg'),quality=92)


def main(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(args.run/'summary.json')
    assert manifest['cases_sha256']==read(framed.REVIEW/'facade-placement-01/verification.json')['cases_sha256']
    index={(r['split'],r['id']):r for r in read(checked(args.run/'cases.json',manifest['cases_sha256']))}
    rank=read(framed.REVIEW/'facade-placement-01/worst/ranking.json')
    selected={r['representative']['id']:r['name'] for r in rank['layouts'] if r['name'] in LABELS}
    jobs=[(s,f,p,index[s,f['id']],args.run,out,selected.get(f['id'])) for s,f,p in framed.cases(framed.EVALUATION)
          if not args.named or f['id'] in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(audit,jobs))
    totals=Counter()
    for r in rows:
        totals.update({k:r['metrics'][k] for k in ('correct','target','placed','extras','missed')})
    if not args.named:
        assert (totals['correct'],totals['extras'],totals['missed'])==(20510,1157,3192)
    for r in rows:
        if 'trials' in r:
            save(out/(r['name'].lower().replace(' ','-')+'-moves.json'),r.pop('trials'))
    save(out/'projection.json',dict(control='028819a',cases_sha256=manifest['cases_sha256'],
                                    sample_stride=STEP,geometry=dict(totals),rows=rows))
    sheet=Image.new('RGB',(1250,348*4))
    for i,name in enumerate(LABELS):
        with Image.open(out/(name.lower().replace(' ','-')+'.jpg')) as panel_image:
            sheet.paste(panel_image,(0,348*i))
    sheet.save(out/'source-comparison.jpg',quality=94)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--out',type=Path,default=REVIEW)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--named',action='store_true')
    main(parser.parse_args())
