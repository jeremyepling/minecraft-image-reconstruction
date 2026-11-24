"""Fit a rendered library, benchmark its atlas and evaluate the fitted grid on touching blocks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine.geometry import UNIT, face, fit_plane, project, transform
from .projection import camera_project, south_quad
from .retrieval import degrade, features, measure, rank

SPLITS=('train','validation','test')
METHODS=('aligned','combined','multiscale')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def capture(root, view, cells, version):
    directory=root/'captures'
    prefix=view['id']
    pose=read(directory/(prefix+'.json'))
    if (pose['minecraft_version']!=version or int(pose['clock_ticks'])%24000!=6000
            or pose['raining'] or pose['thundering'] or not pose['hud_hidden']):
        raise ValueError(f'Invalid capture version, weather, clock or HUD in {prefix}')
    rgb=np.asarray(Image.open(directory/(prefix+'.png')).convert('RGB'))
    if list(rgb.shape[1::-1])!=pose['image_size']:
        raise ValueError(f'Capture dimensions disagree with the saved camera in {prefix}')
    def position(b):
        return b['x'],b['y'],b['z']
    states=read(directory/(prefix+'-blocks.json'))
    actual={position(b):b['block_id'].removeprefix('minecraft:') for b in states}
    if len(actual)!=len(states) or set(actual)!={position(b) for b in cells}:
        raise ValueError(f'Incomplete state readback for {prefix}')
    return rgb,pose,actual


def contiguous_grid(rgb,pose,cells):
    x=min(c['x'] for c in cells)
    y=max(c['y'] for c in cells)+1
    z=cells[0]['z']+1
    nx=max(c['x'] for c in cells)-x+1
    ny=y-min(c['y'] for c in cells)
    if nx*ny!=len(cells):
        raise ValueError('Contiguous calibration panel has missing cells')
    outline=camera_project([(x,y,z),(x+nx,y,z),(x+nx,y-ny,z),(x,y-ny,z)],pose)
    # Known panel outline, approximate block pitch and perturbed phase. The true
    # interior boundaries are withheld from the fitter and used only to score it.
    rough=transform(outline)@np.array([[1/(1.06*nx),0,.31/nx],[0,1/(.94*ny),-.28/ny],[0,0,1]])
    fitted=fit_plane(rgb,dict(name='adjacent wall',outline=outline.tolist()),rough)
    return np.asarray(fitted['grid_to_image'])


def edge_metrics(edges):
    if not edges:
        return dict(count=0)
    values=np.array(edges)
    return dict(count=len(edges),within_tenth_of_block=float(np.mean(values[:,1]<=.1)),
                p50_px=float(np.median(values[:,0])),p95_px=float(np.percentile(values[:,0],95)))


def extract(root,layout):
    pages=defaultdict(list)
    for cell in layout['blocks']:
        pages[cell['page']].append(cell)
    registry={name.removeprefix('minecraft:') for name in read(root/'runtime-block-registry.json')}
    world_positions=set()
    catalogs={s:Counter() for s in SPLITS}
    dataset={key:{s:([],[]) for s in SPLITS} for key in ('block_ids','face_textures')}
    walls={}
    skipped=Counter()
    readbacks=0
    changed=[]
    for view in read(root/'views.json'):
        cells=pages[view['page']]
        split=view['split']
        for c in cells:
            pos=(c['x'],c['y'],c['z'])
            if c['split']!=split or pos in world_positions:
                raise ValueError(f'Overlapping split or reused position at {pos}')
            world_positions.add(pos)
            if view.get('kind')!='adjacent':
                catalogs[split][c['block_id']]+=1
        rgb,pose,actual=capture(root,view,cells,layout['version'])
        readbacks+=len(actual)
        adjacent=view.get('kind')=='adjacent'
        if adjacent:
            h=contiguous_grid(rgb,pose,cells)
            inv=np.linalg.inv(h)
            wall={}
        for c in cells:
            xyz=c['x'],c['y'],c['z']
            name=c['block_id']
            if actual[xyz]!=name:
                skipped['state_changed']+=1
                changed.append(dict(view=view['id'],xyz=xyz,expected=name,actual=actual[xyz]))
                continue
            if c['invisible']:
                skipped['invisible']+=1
                continue
            q=camera_project(south_quad(*xyz),pose)
            if q.min()<0 or np.any(q>=np.array(pose['image_size'])-1):
                skipped['clipped']+=1
                continue
            tile=face(rgb,q,16)
            if adjacent:
                ij=np.floor(project(inv,q.mean(0)))
                pred=project(h,UNIT+ij)
                wall[xyz[:2]]=q,pred,face(rgb,pred,16),c['side_texture']
            else:
                for task,label in [('block_ids',name),('face_textures',c['side_texture'])]:
                    if label:
                        imgs,truth=dataset[task][split]
                        imgs.append(tile)
                        truth.append(label)
        if adjacent:
            walls[view['id']]=(split,wall)
    for split,counts in catalogs.items():
        if set(counts)!=registry or any(n!=1 for n in counts.values()):
            raise ValueError(f'Catalog {split} does not cover the runtime registry once')
    arrays={name:{k:(np.stack(imgs),labels) for k,(imgs,labels) in parts.items()}
            for name,parts in dataset.items()}
    return arrays,walls,dict(registry_ids=len(registry),skipped=skipped,state_changes=changed,readbacks=readbacks)


def train(root,dataset,version,atlas_hash):
    root.mkdir(exist_ok=True)
    tiles,labels=dataset['train']
    expanded=np.concatenate([tiles]+[degrade(tiles,size,(1,1,1)) for size in (6,10,13)])
    ids=np.tile(labels,4)
    model=(features(expanded),ids)
    valid=dataset['validation']
    blurred=degrade(valid[0],9,(1.06,.98,.90)),valid[1]
    clean=measure(model,valid,METHODS)
    soft=measure(model,blurred,METHODS)
    winner=max(METHODS,key=lambda m:(clean[m]['macro']+soft[m]['macro'])/2)
    path=root/'softened-library.npz'
    np.savez_compressed(path,tiles=expanded,labels=ids,method=winner,version=version,
                        atlas_sha256=atlas_hash,task=root.name)
    report=dict(method=winner,validation_clean=clean,validation_soft=soft,
                test=measure(model,dataset['test'],['color',winner],True),
                train_classes=len(set(labels)),known_corner_samples={s:len(set_[1]) for s,set_ in dataset.items()})
    (root/'benchmark.json').write_text(json.dumps(report,indent=1))
    return model,winner,report


def grade_walls(walls,bank,method):
    report={}
    for name,(split,wall) in walls.items():
        groups=defaultdict(list)
        images=[]; labels=[]
        for (x,y),(quad,pred,tile,label) in wall.items():
            if label:
                images.append(tile)
                labels.append(label)
            for dx,dy,edge in ((1,0,[1,2]),(0,-1,[2,3])):
                other=wall.get((x+dx,y+dy))
                if other is None:
                    continue
                err=np.linalg.norm(quad[edge]-pred[edge],axis=1).mean()
                scale=np.linalg.norm(quad[edge[1]]-quad[edge[0]])
                metric=(err,err/scale)
                groups['all'].append(metric)
                groups['same_texture' if other[3]==label else 'different_texture'].append(metric)
        report[name]=dict(split=split,internal_seams={k:edge_metrics(v) for k,v in groups.items()},
                          fitted_crop_material=measure(bank,(np.stack(images),labels),[method])[method])
    return report


def compare_atlas(atlas_path,meta,data,model,method):
    # Use exactly the same eligible opaque face classes in both comparisons.
    rgba=Image.open(atlas_path.parent/meta['atlas']).convert('RGBA')
    known=set(data['train'][1]); labels=[]; tiles=[]
    for texture in meta['textures']:
        label=texture['id'].rsplit('/',1)[-1]
        if label not in known:
            continue
        x,y,w,h=texture['rect']
        sprite=rgba.crop((x,y,x+w,y+h))
        if sprite.getchannel('A').getextrema()[0]!=255:
            continue
        labels.append(label)
        tiles.append(np.asarray(sprite.convert('RGB').resize((16,16),Image.Resampling.BOX)))
    mask=np.isin(data['test'][1],labels)
    test=data['test'][0][mask],np.asarray(data['test'][1])[mask]
    inc=np.isin(model[1],labels)
    shared=(tuple(block[inc] for block in model[0]),model[1][inc])
    return dict(classes=len(labels),samples=len(test[1]),
                atlas=measure((features(np.stack(tiles)),np.array(labels)),test,[method])[method],
                rendered=measure(shared,test,[method])[method])


def evaluate(root,atlas):
    layout=read(root/'layout.json')
    meta=read(atlas)
    if layout['version']!=meta['version'] or layout['client_sha256']!=meta['recipe']['client_sha256']:
        raise ValueError('Atlas and calibration world must come from the same client jar')
    with open(atlas.parent/meta['atlas'],'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=meta['atlas_sha256']:
            raise ValueError('Atlas pixels disagree with metadata')
    data,walls,audit=extract(root,layout)
    result=dict(version=meta['version'],atlas_sha256=meta['atlas_sha256'],audit=audit)
    for name,dataset in data.items():
        bank,method,r=train(root/name,dataset,meta['version'],meta['atlas_sha256'])
        result[name]={k:r[k] for k in ('method','train_classes','known_corner_samples')}
        result[name]['test']={k:{key:v[key] for key in ('macro','correct','n')} for k,v in r['test'].items()}
        if name=='face_textures':
            result['adjacent_walls']=grade_walls(walls,bank,method)
            comparison=compare_atlas(atlas,meta,dataset,bank,method)
            (root/'atlas-comparison.json').write_text(json.dumps(comparison,indent=1))
            result['opaque_shared_classes']=dict(classes=comparison['classes'],samples=comparison['samples'],
                atlas_correct=comparison['atlas']['correct'],rendered_correct=comparison['rendered']['correct'])
    (root/'evaluation.json').write_text(json.dumps(result,indent=1))
    print(json.dumps({k:result[k] for k in ('version','block_ids','face_textures','opaque_shared_classes')},indent=1))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--atlas',type=Path,required=True)
    args=p.parse_args()
    evaluate(args.root,args.atlas)
