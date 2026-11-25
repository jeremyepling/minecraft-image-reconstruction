"""Measure texture recognition and crop membership on held-out multi-angle buildings.

Selection sees development buildings only. Physical repetitions of one angle or lighting
condition count as views of the same building, not independent structures.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from calibration.projection import camera_project, south_quad
from calibration.retrieval import features, distances, exposure_distance
from engine.geometry import UNIT, face, fit_plane, project, transform
from engine.reconstruction import exemplar_bank, geometry

METHODS=('production','exposure','exposure_turns')
CROPS=('known','seeded_grid','image_only')
MATERIALS=('tuff','cobblestone','andesite','deepslate',
           *(f'dead_{s}_coral_block' for s in ('brain','bubble','fire','horn','tube')))


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def clear_rays(positions,eye,starts):
    """Walk each ray from just outside the face until it exits the building bounds."""
    coords=np.array(sorted(positions))
    low=coords.min(0); high=coords.max(0)
    solid=np.zeros(high-low+1,bool)
    solid[tuple((coords-low).T)]=True
    direction=eye-starts
    sign=np.sign(direction).astype(int)
    xyz=np.floor(starts).astype(int)
    delta=np.full(direction.shape,np.inf)
    np.divide(1.,np.abs(direction),out=delta,where=direction!=0)
    boundary=xyz+(sign>0)
    distance=np.full(direction.shape,np.inf)
    np.divide(boundary-starts,direction,out=distance,where=direction!=0)
    visible=np.ones(len(starts),bool)
    active=np.arange(len(starts))
    while len(active):
        active=active[((xyz[active]>=low)&(xyz[active]<=high)).all(1)]
        occupied=solid[tuple((xyz[active]-low).T)]
        visible[active[occupied]]=False
        active=active[~occupied]
        axes=distance[active].argmin(1)
        nearer=distance[active,axes]<1
        active,axes=active[nearer],axes[nearer]
        xyz[active,axes]+=sign[active,axes]
        distance[active,axes]+=delta[active,axes]
    return visible


def area(q):
    return abs(np.dot(q[:,0],np.roll(q[:,1],-1))-np.dot(q[:,1],np.roll(q[:,0],-1)))*.5


def intersection(q,ij):
    points=list(q)
    for axis,lower in ((0,True),(0,False),(1,True),(1,False)):
        limit=ij[axis]+(not lower)
        result=[]
        if not points: return 0.
        a=points[-1]
        for b in points:
            inside_a=(a[axis]>=limit) if lower else (a[axis]<=limit)
            inside_b=(b[axis]>=limit) if lower else (b[axis]<=limit)
            if inside_a!=inside_b:
                result.append(a+(b-a)*((limit-a[axis])/(b[axis]-a[axis])))
            if inside_b: result.append(b)
            a=b
        points=result
    return area(np.array(points)) if len(points)>2 else 0.


def match_cells(candidates, expected, inverse):
    """One-to-one matching at IoU >= 0.5 in block-plane coordinates."""
    pairs=[]; considered=0
    for i,q in enumerate(candidates):
        unwarped=project(inverse,q)
        if not np.isfinite(unwarped).all(): continue
        a=area(unwarped)
        if a<=0: continue
        lo=np.floor(unwarped.min(0)).astype(int); hi=np.ceil(unwarped.max(0)).astype(int)
        if not any(lo[0]<=x<hi[0] and lo[1]<=y<hi[1] for x,y in expected): continue
        considered+=1
        if a<.5 or a>2.: continue
        for x in range(lo[0],hi[0]):
            for y in range(lo[1],hi[1]):
                ij=x,y
                if ij not in expected: continue
                shared=intersection(unwarped,ij)
                iou=shared/(1+a-shared)
                if iou>=.5: pairs.append((iou,i,ij,shared/a))
    matched={}; used=set()
    for iou,i,ij,purity in sorted(pairs,reverse=True):
        if i in used or ij in matched: continue
        matched[ij]=(candidates[i],iou,purity)
        used.add(i)
    return matched,considered


def samples(root,v,blocks,walls,condition,combined=False):
    folder=root/'captures'
    prefix=v['id']; path=folder/(prefix+'.png')
    pose=read(folder/(prefix+'.json'))
    rgb=np.asarray(Image.open(path).convert('RGB'))
    if rgb.shape[1::-1]!=tuple(pose['image_size']): raise ValueError(path)
    coords={(b['x'],b['y'],b['z']):b for b in blocks}
    readback=read(folder/(prefix+'-blocks.json'))
    light={(b['x'],b['y'],b['z']):b['south_sky_light'] for b in readback}
    observed={(b['x'],b['y'],b['z']):b['block_id'].removeprefix('minecraft:') for b in readback}
    if set(coords)!=set(observed) or any(observed[p]!=b['block_id'].removeprefix('minecraft:') for p,b in coords.items()):
        raise ValueError(f'Block readback mismatch in {prefix}')
    # Benchmark only the material-studded vertical facade. Roof and return faces
    # still occlude it and remain distractors for the image-only finder.
    z=walls['z']; left,right,bottom,top=walls['bounds']
    width,height=right-left,top-bottom
    outer=camera_project([[left,top,z+1],[right,top,z+1],[right,bottom,z+1],[left,bottom,z+1]],pose)
    if pose['eye_world'][2]<=z+1: return [],dict(skipped='south facade faces away from camera')
    h=transform(outer)@np.diag([1/width,1/height,1])
    inverse=np.linalg.inv(h)
    expected={}; starts=[]; entries=[]
    clip=Counter()
    u,t=np.meshgrid(np.linspace(.035,.965,5),np.linspace(.035,.965,5))
    for (x,y,bz),b in coords.items():
        if bz!=z or not (left<=x<right and bottom<=y<top): continue
        if not b.get('side_texture'): continue
        ij=x-left,top-y-1
        q=camera_project(south_quad(x,y,z),pose)
        if q.min()<0 or (q>=np.array(pose['image_size'])-1).any():
            clip['clipped']+=1; continue
        entries.append((ij,q,b))
        starts.append(np.column_stack([x+u.ravel(),y+t.ravel(),np.full(u.size,z+1.00001)]))
    if not starts: return [],dict(skipped='no visible front cells')
    live=clear_rays(coords,np.asarray(pose['eye_world']),np.concatenate(starts)).reshape(-1,u.size).all(1)
    for (ij,q,b),seen in zip(entries,live):
        if not seen:
            clip['occluded']+=1;continue
        expected[ij]=q,b
    geom_error=None
    if combined:
        from engine.texel_grid import candidate
        _,cells=candidate(path)
        proposals=[('texel_grid',[np.asarray(c['quad']) for c in cells])]
    else:
        rough=transform(outer)@np.array([[1/(1.06*width),0,.31/width],[0,1/(.94*height),-.28/height],[0,0,1]])
        fitted=fit_plane(rgb,dict(name='benchmark facade',outline=outer.tolist()),rough)
        seed=[np.asarray(c['quad']) for c in fitted['cells']]
        try:
            _,_,pred=geometry(path)
            pred=[np.asarray(c['quad']) for c in pred]
        except ValueError as exc:
            pred=[];geom_error=str(exc)
        proposals=[('seeded_grid',seed),('image_only',pred)]
    maps={}; coverage={}
    for name,q in proposals:
        maps[name],considered=match_cells(q,expected,inverse)
        coverage[name]=dict(generated=len(q),intersecting_wall_boxes=considered,matched=len(maps[name]))
        # A proposed full wall cell over an actual aperture is a separate failure
        # from an incorrectly classified occupied cell. Mask apertures obscured
        # by a roof before counting proposals against them.
        holes=[]
        for x in range(left,right):
            for y in range(bottom,top):
                if (x,y,z) in coords: continue
                hole_quad=camera_project(south_quad(x,y,z),pose)
                if hole_quad.min()>=0 and (hole_quad<np.array(pose['image_size'])-1).all():
                    holes.append((x-left,top-y-1))
        if holes:
            hole_rays=np.concatenate([np.column_stack([left+i+u.ravel(),top-j-1+t.ravel(),
                                                       np.full(u.size,z+1.00001)]) for i,j in holes])
            visible=clear_rays(coords,np.asarray(pose['eye_world']),hole_rays).reshape(len(holes),-1).all(1)
            visible_holes={ij for ij,ok in zip(holes,visible) if ok}
            false_cells,_=match_cells(q,visible_holes,inverse)
            coverage[name].update(visible_holes=len(visible_holes),covered_holes=len(false_cells))
    if not combined:
        maps['known']={ij:(q,1.,1.) for ij,(q,_) in expected.items()}
    rows=[]
    for ij,(q,b) in expected.items():
        x,y=b['x'],b['y']
        adjacent=set()
        for nx,ny in ((x-1,y),(x+1,y),(x,y+1),(x,y-1)):
            if not (left<=nx<right and bottom<=ny<top): continue
            neighbor=coords.get((nx,ny,z))
            if neighbor is None: adjacent.add('opening')
            elif neighbor['side_texture']==b['side_texture']: adjacent.add('same')
            else: adjacent.add('unlike')
        shade='overhang' if any(px==x and py>y and pz>z for px,py,pz in coords) else 'open_sky'
        scale=float(np.sqrt(area(q)))
        for name,mapping in maps.items():
            match=mapping.get(ij)
            values=dict(view=prefix,condition=condition,angle=v.get('angle'),crop=name,texture=b['side_texture'],
                        xyz=(x,y,z),neighbors=sorted(adjacent),shade=shade,px=scale,sky=light[x,y,z])
            if match is not None:
                rect,iou,purity=match
                values.update(tile=face(rgb,rect,16),iou=iou,purity=purity)
            rows.append(values)
    return rows,dict(visible=len(expected),excluded=clip,geometry_error=geom_error,crops=coverage)

def library_ranks(queries,train,labels,method,variants):
    """All candidates compete; exposure fit is bounded and pays a log-gain penalty."""
    order=np.argsort(labels,kind='stable')
    ids,starts=np.unique(labels[order],return_index=True)
    guesses={m:[] for m in variants}
    for start in range(0,len(queries),48):
        tiles=queries[start:start+48]
        turns=4 if 'exposure_turns' in variants else 1
        best=None
        for k in range(turns):
            q=features(np.rot90(tiles,k,axes=(1,2)))
            d=distances(q,train)[method]
            if k==0 and 'production' in variants:
                guesses['production'].extend(ids[np.minimum.reduceat(d[:,order],starts,axis=1).argmin(1)])
            if not any(v!='production' for v in variants): continue
            # Side-face illumination can darken all channels together. Fit one
            # gain per candidate, preserving chromatic differences and texture.
            moderated=exposure_distance(q,train,method,d)
            if k==0 and 'exposure' in variants:
                guesses['exposure'].extend(ids[np.minimum.reduceat(moderated[:,order],starts,axis=1).argmin(1)])
            if 'exposure_turns' in variants:
                best=moderated if best is None else np.minimum(best,moderated+.018)
        if 'exposure_turns' in variants:
            guesses['exposure_turns'].extend(ids[np.minimum.reduceat(best[:,order],starts,axis=1).argmin(1)])
    return guesses


def score(rows,method):
    confusion=defaultdict(Counter)
    groups=defaultdict(Counter)
    unique=set()
    for r in rows:
        matched='tile' in r
        pred=r.get('pred',{}).get(method,'<missed>')
        truth=r['texture']
        confusion[truth][pred]+=1
        unique.add(r['xyz'])
        keys=['all','condition/'+r['condition'],'exposure/'+r['shade'],
              'skylight/'+str(r['sky']),'angle/'+str(r['angle']),
              'size/'+('under_10px' if r['px']<10 else 'at_least_10px')]
        keys+=['neighbor/'+name for name in r['neighbors']]
        for key in keys:
            c=groups[key]
            c['n']+=1;c['matched']+=int(matched);c['correct']+=int(pred==truth)
    edges=defaultdict(Counter)
    by_pos={(r['condition'],r['view'],r['xyz']):r for r in rows}
    for row in rows:
        x,y,z=row['xyz']
        for other_pos in [(x+1,y,z),(x,y+1,z)]:
            other=by_pos.get((row['condition'],row['view'],other_pos))
            if other is None: continue
            a,b=row['texture'],other['texture']
            p=row.get('pred',{}).get(method)
            q=other.get('pred',{}).get(method)
            for key in ['same' if a==b else 'unlike',' / '.join(sorted((a,b)))]:
                c=edges[key];c['n']+=1
                c['both_matched']+=int(p is not None and q is not None)
                c['both_correct']+=int((a,b)==(p,q))
                c['different_called_same']+=int(a!=b and p is not None and p==q)
    return dict(unique_blocks=len(unique),counts=dict(groups),confusion=dict(confusion),
                ordered_positions_per_neighbor_pair=dict(edges))


def objective(rows,m):
    # Give each appearance condition and material equal weight at selection.
    hits=defaultdict(list)
    for r in rows:
        if r['crop']=='known':
            hits[r['condition'],r['texture']].append(r['pred'][m]==r['texture'])
    return float(np.mean([np.mean(vals) for vals in hits.values()]))


def extract_split(root,conditions,split,combined=False):
    pages=defaultdict(list)
    layout=read(root/'layout.json')
    buildings={b['id']:b for b in read(root/'buildings.json')}
    for b in layout['blocks']:
        if b.get('kind')=='adjacency' and b['split']==split: pages[b['page']].append(b)
    rows=[]; cover={}
    for condition in conditions:
        base=root if condition=='normal' else root/'conditions'/condition
        views=read(base/'views.json')
        candidates=[v for v in views if v['page'] in pages and v['split']==split]
        if not candidates: raise ValueError(f'Missing {condition} views for {split}')
        for v in candidates:
            blocks=pages[v['page']]
            wall=[b for b in blocks if b['role']=='adjacency_wall']
            x,y,z=zip(*[(b['x'],b['y'],b['z']) for b in wall])
            if len(set(z))!=1: raise ValueError('Adjacency walls must be coplanar')
            apertures=[np.floor(p['point']).astype(int) for p in buildings[v['building_id']]['probes']
                       if p['role']=='air']
            xs=[*x,*(p[0] for p in apertures)]
            ys=[*y,*(p[1] for p in apertures)]
            walls=dict(z=z[0],bounds=(min(xs),max(xs)+1,min(ys),max(ys)+1))
            batch,audit=samples(base,v,blocks,walls,condition,combined)
            rows.extend(batch)
            cover[f"{condition}/{v['id']}"]=audit
    return rows,cover


def run(args):
    lib=args.library; out=args.out
    out.mkdir(parents=True,exist_ok=True)
    catalog=read(args.atlas/'placement-index.json')
    bank,labels,method=exemplar_bank(args.atlas,lib,catalog)
    include=np.isin(labels,list(catalog['choices']['south']))
    train=features(bank[include]); labels=labels[include]
    conditions=args.conditions
    signature=dict(version=catalog['version'],atlas_sha256=catalog['atlas_sha256'],
                   library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                   fixture_sha256=hashlib.sha256((args.root/'layout.json').read_bytes()).hexdigest(),
                   classifier=method,conditions=conditions)
    if args.stage=='select':
        split='dev'; variants=METHODS
    else:
        prior=read(out/'selection.json')
        if prior['signature']!=signature: raise ValueError('Held-out configuration differs from selection')
        split='test'; variants=list(dict.fromkeys(['production',prior['selected']]))
    combined=args.stage=='combined'
    rows,coverage=extract_split(args.root,conditions,split,combined)
    missing=sorted(set(MATERIALS)-{r['texture'] for r in rows})
    if missing: raise ValueError(f'Missing required texture classes: {missing}')
    found=[r for r in rows if 'tile' in r]
    preds=library_ranks(np.stack([r['tile'] for r in found]),train,labels,method,variants) if found else {}
    for i,row in enumerate(found):
        row['pred']={m:str(v[i]) for m,v in preds.items()}
    reports={}
    for crop in (('texel_grid',) if combined else CROPS):
        selected=[r for r in rows if r['crop']==crop]
        quality=[r for r in selected if 'tile' in r]
        reports[crop]=dict(expected=len(selected),matched=len(quality),
                           mean_iou=np.mean([r['iou'] for r in quality]).item() if quality else None,
                           mean_purity=np.mean([r['purity'] for r in quality]).item() if quality else None,
                           methods={m:score(selected,m) for m in variants})
    result=dict(signature=signature,split=split,coverage=coverage,results=reports)
    if combined:
        frozen=read(out/'heldout.json')
        if frozen['signature']!=signature: raise ValueError('Different original benchmark signature')
        result['original_image_only']={m:s['counts'] for m,s in frozen['results']['image_only']['methods'].items()}
        for m,old in result['original_image_only'].items():
            new=reports['texel_grid']['methods'][m]['counts']
            if any(new[k]['n']!=v['n'] for k,v in old.items()):
                raise ValueError('Expected held-out observations differ')
    if split=='dev':
        scores={m:objective(rows,m) for m in variants}
        result.update(selected=max(scores,key=scores.get),objective=scores)
        print('Selected',result['selected'],scores)
    name='combined' if combined else 'selection' if split=='dev' else 'heldout'
    (out/f'{name}.json').write_text(json.dumps(result,indent=1))
    for crop,entry in reports.items():
        print(crop,{m:s['counts']['all'] for m,s in entry['methods'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['select','test','combined'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--atlas',type=Path,required=True,help='Versioned atlas and placement-index directory')
    p.add_argument('--conditions',nargs='+',default=['normal','small','warm-shader'])
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
