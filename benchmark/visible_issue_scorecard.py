"""Score photographed regions reported by Jeremy using fixed, evaluation-only pixel probes."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as frozen
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked,digest,read,save
from benchmark.texture_camera_sheet import font
from benchmark.wall_source_projection import STEP,Scene
from benchmark.wall_texture_masks import native_mask,rewrite

ROOT=Path('benchmark/reviews')
RANK=ROOT/'facade-placement-01/worst/ranking.json'
CONTACT=ROOT/'wall-section-01/contacts/contacts.json'
FLOOR=ROOT/'wall-source-projection-01/floor-controls.json'
NAMES=('Canopy 04','Colonnade 01','Dev Relief 00','Relief 00','Shopfront','Facade')


def index(folder):
    meta=read(folder/'summary.json')
    return {r['id']:r for r in read(checked(folder/'cases.json',meta['cases_sha256']))}


def regions(name,key,photo,parts,camera,labels,ids,base,missed):
    target=ids[::STEP,::STEP].ravel()
    if name in ('Dev Relief 00','Relief 00'):
        contacts=next(r for r in read(CONTACT)['frames'] if r['id']==key)
        for i,row in enumerate(contacts['illustrated'],1):
            xyz=row['xyz']
            targets=[f['id'] for f in labels['faces'] if f['xyz_world']==xyz]
            yield f'relief detail {i}',np.isin(target,targets)
        return
    if name=='Colonnade 01':
        case=next(r for r in read(FLOOR)['measured'] if r['id']==key)
        cells={tuple(v['cell']) for v in case['cells']}
        yield 'pavement visible through openings',(target==0)&np.isin(base.owner,[i for i,c in enumerate(base.cells) if c in cells])
        walls={tuple(f['xyz_world']) for f in labels['faces'] if not f['normal_world'][1]}
        faces=[f['id'] for f in labels['faces'] if f['normal_world'][1]==1 and tuple(f['xyz_world']) in (missed-walls)]
        yield 'visible missed top faces',np.isin(target,faces)
        return
    mode={'Canopy 04':'pillar','Facade':'diagonal','Shopfront':'shade'}[name]
    edited,_=rewrite(photo,parts,camera,mode)
    old={r['id']:r for r in parts}
    new=[r for r in edited if r['id'] not in old]
    if name=='Canopy 04':
        # Each sibling is separately scored, so the large foreground post cannot
        # overwhelm a depth error in the thin strip beside it.
        masks=sorted((native_mask(p,photo.size) for p in new),key=lambda m:int(m.sum()))
        for i,mask in enumerate(masks):
            yield ('narrow pillar strip' if i==0 else 'wide pillar face'),mask[::STEP,::STEP].ravel()
    elif name=='Facade':
        merged=np.logical_or.reduce([native_mask(p,photo.size) for p in new])
        yield 'wall crossed by diagonal',merged[::STEP,::STEP].ravel()
    else:
        before=np.logical_or.reduce([native_mask(p,photo.size) for p in parts])
        after=np.logical_or.reduce([native_mask(p,photo.size) for p in edited])
        band=(after&~before)[::STEP,::STEP].ravel()
        yield 'lower edge on build',band&(target!=0)
        yield 'lower edge on background',band&(target==0)


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    control=index(args.control)
    candidate=index(args.run)
    named={r['representative']['id']:r['name'] for r in read(RANK)['layouts'] if r['name'] in NAMES}
    results=[]
    thumbs={}
    for split,frame,inputs in frozen.cases(frozen.EVALUATION):
        key=frame['id']
        if key not in named:
            continue
        row,other=control[key],candidate[key]
        for attr in ('rgb_sha256','input_hashes','reference_hashes'):
            assert row[attr]==other[attr]
        photo,parts,_,cam,pose,labels,ids,truth,_,placed,_=read_view(args.control,split,frame,inputs,row)
        def scene(folder,line):
            return Scene(photo,parts,cam,set(decoded_pack(checked(folder/split/key/'reconstruction.zip',line['pack_sha256']))))
        base=scene(args.control,row)
        current=base if row['pack_sha256']==other['pack_sha256'] else scene(args.run,other)
        coord=np.zeros((len(labels['faces'])+1,3),int)
        for f in labels['faces']:
            coord[f['id']]=f['xyz_world']
        target=ids[::STEP,::STEP].ravel()
        def matches(test):
            registered=registration(test.P,pose)
            if not test.cells:
                return target==0
            points=np.floor((np.array(test.cells)+.5)@registered[:3,:3].T+registered[:3,3]).astype(int)
            return ((test.owner<0)&(target==0))|((test.owner>=0)&(target>0)&(points[test.owner]==coord[target]).all(1))
        old,new=matches(base),matches(current)
        name=named[key]
        pic=np.asarray(photo)[::STEP,::STEP,:3].copy()
        masks=[]
        details=list(regions(name,key,photo,parts,cam,labels,ids,base,truth-placed))
        for title,mask in details:
            n=int(mask.sum())
            assert n>0
            shaped=mask.reshape(base.shape)
            outline=np.asarray(Image.fromarray(shaped).convert('L'))
            edge=(outline!=np.roll(outline,1,0))|(outline!=np.roll(outline,1,1))
            pic[edge]=[48,238,220]
            masks.append(dict(region=title,samples=n,before=int(old[mask].sum()),after=int(new[mask].sum())))
        thumb=ImageOps.contain(Image.fromarray(pic),(380,228))
        thumbs[name]=thumb
        results.append(dict(id=key,name=name,rgb_sha256=row['rgb_sha256'],previous_pack=row['pack_sha256'],pack=other['pack_sha256'],
                            baseline_f1=row['f1'],f1=other['f1'],probes=masks))
        if args.viewer:
            from benchmark.visible_issue_viewer import layers
            results[-1]['view']=layers(args.out/'layers',key,photo,details,old,new,base.owner>=0,current.owner>=0)
    assert len(results)==len(NAMES)
    plate=Image.new('RGB',(1180,560),'#102731')
    ink=ImageDraw.Draw(plate)
    ink.text((12,8),'Photographed regions checked across placement experiments',font=font(24),fill='white')
    for i,name in enumerate(NAMES):
        x,y=12+390*(i%3),52+252*(i//3)
        ink.text((x,y),name,font=font(18),fill='white')
        plate.paste(thumbs[name],(x,y+26))
    plate.save(args.out/'regions.jpg',quality=93)
    save(args.out/'cases.json',dict(step=STEP,source_sha256={p.name:digest(p) for p in (RANK,CONTACT,FLOOR)},
                                  baseline_cases_sha256=digest(args.control/'cases.json'),
                                  scored_cases_sha256=digest(args.run/'cases.json'),frames=results))
    if args.viewer:
        from benchmark.visible_issue_viewer import page
        page(args.out,sorted(results,key=lambda r:NAMES.index(r['name'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,default=Path('../wall-section-round/runs/control'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--viewer',action='store_true')
    run(parser.parse_args())
