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
from engine.projected_appearance import FaceRenderer as _FaceRenderer, Scene, STEP

RUN=Path('../facade-placement-round/runs/production')
REVIEW=Path('benchmark/reviews/wall-source-projection-01')
LABELS=('Colonnade 01','Dev Relief 00','Relief 00','Facade')

class FaceRenderer(_FaceRenderer):
    def __init__(self,scene):
        super().__init__(scene,_atlas('26.3',STABLE/'runs/26.3/atlas'))


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
    if name:
        missed=set(truth)-scored
        # Attribute each sampled ray to a labeled missing block, then measure how a
        # render that lacks that block can nevertheless cover the same camera rays.
        lut=np.zeros(len(faces)+1,int)
        for number,face in enumerate(faces,1):
            assert face['id']==number
            lut[number]=int(tuple(face['xyz_world']) in missed)
        missing=lut[truth_sample]>0
        stats['missing_block_samples']=int(missing.sum())
        stats['missing_samples_covered_by_other_blocks']=int((missing&predicted).sum())
        stats['missing_samples_other_block_same_orientation']=int((missing&matched_axis).sum())
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
    # Benchmark exports place smooth stone. Keep the unplaced background neutral
    # so source pixels cannot visually fill a hole in either synthetic render.
    stone=rend.atlas.tiles[rend.atlas.labels.index('smooth_stone')].reshape(256,3)
    shape=np.full((len(pred),3),42,np.uint8)
    for n,idx in s.owned.items():
        for a in np.unique(s.axes[idx]):
            face=idx[s.axes[idx]==a]
            shade_factor=(.64,1.,.82)[a] if a!=1 or s.eye[1]>s.cells[n][1]+1 else .52
            shape[face]=stone[s.uv(face,s.cells[n],a)]*shade_factor
    shade=np.full_like(rend.source,42)
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
    tiles=[(src,'Source crop'),(Image.fromarray(shape.reshape(h,w,3)),'Export in smooth stone'),
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
