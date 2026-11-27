"""Read projected atlas rows from the two sides of a photographed pillar seam."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from benchmark.wall_texture_masks import native_mask, orientation, rewrite
from engine.camera_atlas_floor import _atlas
from benchmark.framed_geometry_order import ATLAS
from benchmark.mask_blocks import decoded_pack
from engine.thin_atlas_faces import _match, _mips, _corr

CONTROL=Path('../wall-section-round/runs/control')
CANOPY='shape-benchmark-26.3--train-canopy-04-left'


@lru_cache(maxsize=1)
def atlas_maps():
    atlas=_atlas('26.3',ATLAS)
    return atlas,_mips(atlas)


def sample_mask(region,size):
    mask=native_mask(region,size)
    padded=np.pad(mask,1)
    inner=mask & padded[1:-1,:-2] & padded[1:-1,2:] & padded[:-2,1:-1] & padded[2:,1:-1]
    y,x=np.nonzero(inner)
    if len(x)>4000:
        take=np.random.default_rng(5681).choice(len(x),4000,replace=False)
        y,x=y[take],x[take]
    return y,x


def predict(image,region,camera,cells):
    y,x=sample_mask(region,image.size)
    if len(x)<60:
        return None
    p,rotation=orientation(camera,image.size)
    axis=int(np.argmax(abs(rotation.T@region['normal_camera'])))
    if axis==1:
        return None
    free=[i for i in range(3) if i!=axis]
    yi=free.index(1)
    inv=np.linalg.inv(p[:,:3])
    eye=-inv@p[:,3]
    rays=np.column_stack((x+.5,y+.5,np.ones(len(x))))@inv.T
    rgb=np.asarray(image.convert('RGB'))[y,x].astype(float)
    atlas,maps=atlas_maps()
    levels=[q[axis] for q in cells]
    plans=[]
    for level in range(min(levels)-3,max(levels)+4):
        plane=level+int(eye[axis]>level+1)
        if level<=eye[axis]<=level+1 or np.min((plane-eye[axis])/rays[:,axis])<=0:
            continue
        xyz=eye+(plane-eye[axis])*rays/rays[:,[axis]]
        texel=np.floor(xyz[:,free]*16).astype(int)
        ijk=np.floor(xyz).astype(int)
        ijk[:,axis]=level
        if np.quantile(ijk[:,1],.98)-np.quantile(ijk[:,1],.02)<1:
            continue
        phase=texel%16
        keys=16*phase[:,1]+phase[:,0]
        folds=ijk[:,1]%2
        metrics=[]
        for fold in (0,1):
            seen=folds==fold
            if min(seen.sum(),(~seen).sum())<32:
                break
            # Fit a sprite only on alternating block courses, then compare
            # the unused courses with its projected texels at that depth.
            fit=_match(texel[seen],rgb[seen],atlas,maps)
            if not fit:
                break
            tested=~seen
            unique,groups,n=np.unique(keys[tested],return_inverse=True,return_counts=True)
            if len(unique)<10:
                break
            gray=np.bincount(groups,weights=rgb[tested].mean(1))/n
            template=maps[fit['sprite'],fit['mip'],fit['turn']]
            metrics.append(float(_corr(template[unique],gray)))
        if len(metrics)==2:
            plans.append((min(metrics),level,ijk,texel[:,yi]))
    if not plans:
        return None
    plans.sort(key=lambda v:v[0],reverse=True)
    best,level,ijk,tex=plans[0]
    margin=best-plans[1][0] if len(plans)>1 else 0.
    accepted=best>=.77 and margin>=.12
    return dict(y=y,x=x,score=best,margin=margin,level=level,ijk=ijk,
                plans=plans,accepted=accepted,tex=tex,axis=axis)


def score_job(job):
    s,frame,inputs,row,out=job
    image,parts,_,cam=framed.photo(s,frame,inputs)
    pieces,ops=rewrite(image,parts,cam,'pillar')
    record=dict(split=s,id=frame['id'],rgb=row['rgb_sha256'],pack=row['pack_sha256'],parts=[])
    if not ops:
        return record
    prior=set(decoded_pack(checked(CONTROL/s/frame['id']/'reconstruction.zip',row['pack_sha256'])))
    if not prior:
        return record
    new_ids={p['id'] for p in pieces}-{p['id'] for p in parts}
    predictions=[(piece,predict(image,piece,cam,prior)) for piece in pieces if piece['id'] in new_ids]
    # Coordinates and surface IDs are opened after all image-only predictions.
    source='old' if s in ('new','extra') else s
    builds=framed.NEW_INPUTS if s=='new' else framed.transfer.BUILDS
    truth,_,t,_=framed.references(source,frame,framed.NATIVE_INPUTS.get(s),framed.TRAINING_DATA,
                                  framed.DEPTH,dict(native=framed.native.BUILDS,old=builds,fresh=builds))
    if s in ('new','extra'):
        from training.visible_labels import visible_labels
        path=builds/frame['directory']/'captures'/frame['source_id']
        hashes=frame['files']
        pose=read(checked(path.with_suffix('.json'),hashes['camera.json']))
        blocks=read(checked(path.with_name(path.name+'-blocks.json'),hashes['world-readback.json']))
        labels,ids,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in blocks},(0,0,0))
    else:
        _,_,_,_,_,labels,ids,*_=read_view(CONTROL,s,frame,inputs,row)
    coords=np.array([[-999]*3]+[f['xyz_world'] for f in sorted(labels['faces'],key=lambda v:v['id'])])
    def world(ijk):
        return np.floor((ijk+.5)@t[:3,:3].T+t[:3,3]).astype(int)+row['translation']
    panels=[]
    for piece,pred in predictions:
        if pred is None:
            record['parts'].append(dict(id=piece['id'],status='insufficient'))
            continue
        x,y=pred['x'],pred['y']
        actual=coords[ids[y,x]]
        assigned=world(pred['ijk'])
        def fraction(plan):
            return float((world(plan)==actual).all(1).mean())
        truth_votes=[fraction(v[2]) for v in pred['plans']]
        best_index=int(np.argmax(truth_votes))
        values,n=np.unique(pred['ijk'],axis=0,return_counts=True)
        proposed={tuple(v) for v,num in zip(values,n) if num>=8}
        new=proposed-prior
        new_world=set(map(tuple,world(np.array(sorted(new))))) if new else set()
        record['parts'].append(dict(id=piece['id'],status='accepted' if pred['accepted'] else 'abstained',
                     axis=pred['axis'],pixels=len(x),score=round(pred['score'],4),margin=round(pred['margin'],4),
                     predicted_level=pred['level'],best_available_level=pred['plans'][best_index][1],
                     correct_pixel_fraction=round(fraction(pred['ijk']),4),best_possible_fraction=round(max(truth_votes),4),
                     observed_courses=len(set(pred['ijk'][:,1])),proposed_cells=len(proposed),
                     additional_visible=len(new_world&truth),additional_wrong=len(new_world-truth)))
        if frame['id']==CANOPY:
            full=np.array(image)
            mark=(pred['tex']%16<=1)
            full[y[mark],x[mark]]=[0,246,213]
            bounds=native_mask(piece,image.size).nonzero()
            y0,x0=max(0,min(bounds[0])-8),max(0,min(bounds[1])-12)
            y1,x1=min(image.height,max(bounds[0])+9),min(image.width,max(bounds[1])+13)
            panels.append((Image.fromarray(full).crop((x0,y0,x1,y1)),record['parts'][-1]))
    if panels:
        plate=Image.new('RGB',(840,430),'#173340')
        pen=ImageDraw.Draw(plate)
        pen.text((12,9),'Canopy 04  |  Reading block courses on either side of the seam',font=font(21),fill='white')
        plate.paste(ImageOps.contain(image,(390,280)),(10,77))
        for k,(crop,q) in enumerate(panels):
            xx=411+209*k
            crop=ImageOps.contain(crop,(193,298),Image.Resampling.NEAREST)
            plate.paste(crop,(xx,45))
            pen.text((xx,353),f"{q['observed_courses']} sampled rows",font=font(17),fill='white')
            pen.text((xx,382),f"{100*q['correct_pixel_fraction']:.0f}% exact block hits",font=font(16),fill='#abd8d3')
        plate.save(out/'canopy-04.jpg',quality=94)
    return record


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    meta=read(CONTROL/'summary.json')
    rows={(r['split'],r['id']):r for r in read(checked(CONTROL/'cases.json',meta['cases_sha256']))}
    jobs=[(s,f,h,rows[s,f['id']],args.out) for s,f,h in framed.cases(args.splits)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(score_job,jobs))
    save(args.out/'rows.json',dict(control_cases_sha256=meta['cases_sha256'],frames=records))
    for r in records:
        if r['parts']:
            print(r['id'],[(p['status'],p.get('correct_pixel_fraction'),p.get('additional_visible')) for p in r['parts']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits',nargs='+',choices=framed.SIZES,default=list(framed.SIZES))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args())
