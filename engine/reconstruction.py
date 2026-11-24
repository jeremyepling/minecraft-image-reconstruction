"""Infer geometry first, rank surfaces second, and export deterministic block placements."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from calibration.retrieval import features, rank, degrade
from engine.atlas import digest
from engine.geometry import UNIT, face, project, transform, fit_plane, detect
from engine.dimensions import measure
from engine.lighting import cast_estimate, correct
from engine.propose import silhouette, estimate_grid
from engine.export import write_pack

RIGHT={'south':(1,0,0),'north':(-1,0,0),'west':(0,0,1),'east':(0,0,-1),
       'up':(1,0,0),'down':(1,0,0)}
ROW_UP={'up':(0,0,-1),'down':(0,0,1)}


def geometry(image_path, scene=None):
    with Image.open(image_path) as f:
        image=ImageOps.exif_transpose(f).convert('RGBA')
    rgb=np.asarray(image)[:,:,:3]
    alpha=np.asarray(image)[:,:,3]
    fitted_planes={}
    hints={}
    if scene:
        hints=json.loads(Path(scene).read_text())
        if hints.get('image_sha256',digest(image_path))!=digest(image_path):
            raise ValueError('Scene belongs to a different image')
        planes=hints['planes']
        if 'directions' in hints:
            fitted_planes={p['name']:p for p in detect(rgb,hints)}
    else:
        outlines,foreground=silhouette(image)
        if foreground is not None:
            alpha=np.minimum(alpha,foreground)
        planes=[dict(name=f'wall-{i+1}',quad=q) for i,q in enumerate(outlines)]
        if len(planes)==2: planes[0]['facing']='west'
    if not planes or len({s['name'] for s in planes})!=len(planes):
        raise ValueError('Provide at least one plane and give each a unique name')
    cells=[]; measured=[]; previous_width=0
    for index,p in enumerate(planes):
        quad=np.array(p.get('quad',p.get('outline')),float)
        if quad.shape!=(4,2) or not np.isfinite(quad).all():
            raise ValueError('Expected four finite image corners, clockwise from top left')
        if (quad<0).any() or (quad>image.size).any():
            raise ValueError('Plane lies outside the input image')
        raw_grid=transform(quad)
        known='columns' in p and 'rows' in p
        if ('columns' in p)!=('rows' in p):
            raise ValueError('Supply rows and columns together')
        seeded=p['name'] in fitted_planes
        if seeded:
            source=fitted_planes[p['name']]
            h=np.array(source['grid_to_image'])
            rectified=project(np.linalg.inv(h),quad)
            lo=np.rint(rectified.min(0)).astype(int)
            cols,rows=map(int,np.maximum(1,np.rint(rectified.max(0)).astype(int)-lo))
            strength=[a[0] for a in source['axes']]
        elif known:
            cols,rows=p['columns'],p['rows']
            strength=None
        else:
            (cols,sc),(rows,sr)=estimate_grid(rgb,quad)
            strength=[sc,sr]
        if any(type(v)!=int or v<1 for v in (cols,rows)) or cols*rows>65536:
            raise ValueError('Plane needs positive integer counts and at most 65536 cells')
        if seeded:
            h=h@np.array([[1,0,lo[0]],[0,1,lo[1]],[0,0,1]])
        else:
            h=raw_grid@np.diag([1/cols,1/rows,1])
        # Fit the approved periodic seam model when counts are estimated. Keep
        # measured edges when an integer wall extent was explicitly supplied.
        if not seeded and not known and min(cols,rows)>=3 and cols*rows<=8192:
            fitted=fit_plane(rgb,dict(name=p['name'],outline=quad.tolist(),holes=p.get('holes',[])),h,metric=True)
            h=np.array(fitted['grid_to_image'])
            rectified=project(np.linalg.inv(h),quad)
            lo=np.rint(rectified.min(0)).astype(int)
            hi=np.rint(rectified.max(0)).astype(int)
            cols,rows=map(int,np.maximum(1,hi-lo))
            h=h@np.array([[1,0,lo[0]],[0,1,lo[1]],[0,0,1]])
        facing=p.get('facing','south')
        if facing not in RIGHT: raise ValueError(f'Unsupported face direction: {facing}')
        origin=p.get('origin')
        if origin is None and seeded and 'dimensions' in hints:
            frame=hints['dimensions']['front_elevation']
            inv=np.linalg.inv(fitted_planes[frame['reference_plane']]['grid_to_image'])
            xy=project(inv,quad[3])-project(inv,frame['origin_px'])
            origin=[round(float(xy[0])),round(float(-xy[1])),0]
        if origin is None:
            if scene and index: raise ValueError('Each additional scene wall needs its origin')
            origin=[0,0,previous_width-1] if index else [0,0,0]
        if len(origin)!=3 or any(type(a)!=int for a in origin): raise ValueError('origin needs three integer coordinates')
        mask=Image.fromarray(alpha.copy())
        draw=ImageDraw.Draw(mask)
        for hole in p.get('holes',[]): draw.polygon([tuple(x) for x in hole],fill=0)
        valid=np.asarray(mask)
        for j in range(rows):
            for i in range(cols):
                corners=project(h,UNIT+[i,j])
                cx,cy=np.floor(corners.mean(0)).astype(int)
                if not (0<=cx<image.width and 0<=cy<image.height) or valid[cy,cx]<128: continue
                pos=np.array(origin)+i*np.array(RIGHT[facing])+(rows-1-j)*np.array(ROW_UP.get(facing,(0,1,0)))
                cells.append(dict(plane=p['name'],ij=[i,j],quad=corners.round(4).tolist(),
                                  xyz=pos.tolist(),facing=facing))
        measured.append(dict(name=p['name'],quad=quad.tolist(),columns=cols,rows=rows,origin=origin,
                             facing=facing,grid_to_image=h.tolist(),spacing_score=strength))
        previous_width=cols
    if not cells: raise ValueError('No visible cells in the supplied image/scene')
    return rgb,measured,cells


def analyze(image,scene,out):
    rgb,planes,cells=geometry(image,scene)
    geometry_data=dict(image_sha256=digest(image),planes=planes,cells=cells)
    out.mkdir(parents=True,exist_ok=True)
    (out/'geometry.json').write_text(json.dumps(geometry_data))
    limits=[[min(v),max(v)] for v in zip(*(c['xyz'] for c in cells))]
    dimensions=dict(planes=planes,observed_bounds_xyz=limits,observed_size_xyz=[hi-lo+1 for lo,hi in limits],
                    source='scene' if scene else 'automatic proposal')
    if scene:
        hints=json.loads(Path(scene).read_text())
        if 'dimensions' in hints:
            outlines=[dict(p,outline=p.get('quad',p.get('outline'))) for p in hints['planes']]
            by_plane=defaultdict(list)
            for c in cells: by_plane[c['plane']].append(dict(c,visible_fraction=1.))
            fitted=dict(geometry_data,planes=[dict(p,cells=by_plane[p['name']]) for p in planes])
            dimensions['landmark_measurements']=measure(fitted,dict(hints,planes=outlines))
    (out/'dimensions.json').write_text(json.dumps(dimensions,indent=2))
    if scene is None:
        fields=('name','quad','columns','rows','origin','facing')
        proposal=dict(image_sha256=geometry_data['image_sha256'],
                      planes=[{key:p[key] for key in fields} for p in planes])
        (out/'proposed-scene.json').write_text(json.dumps(proposal,indent=2))
    preview=Image.fromarray(rgb); pen=ImageDraw.Draw(preview)
    for c in cells:
        q=[tuple(pt) for pt in c['quad']];pen.line(q+[q[0]],fill=(42,203,183),width=1)
    preview.thumbnail((1600,1200));preview.save(out/'grid.png')
    return rgb,planes,cells


def exemplar_bank(atlas_dir, library, catalog):
    atlas=json.loads((atlas_dir/'atlas.json').read_text())
    png=atlas_dir/atlas['atlas']
    if digest(png)!=atlas['atlas_sha256'] or (atlas['atlas_sha256'],atlas['version'])!=(
                catalog['atlas_sha256'],catalog['version']):
        raise ValueError('Atlas, placement index or target version mismatch')
    if library:
        with np.load(library,allow_pickle=False) as f:
            if 'task' in f and str(f['task'])!='face_textures':
                raise ValueError('Use the face_textures calibration library for surface reconstruction')
            if str(f['version'])!=atlas['version'] or str(f['atlas_sha256'])!=atlas['atlas_sha256']:
                raise ValueError('Calibration version or atlas digest mismatch')
            return f['tiles'],f['labels'],str(f['method'])
    images=[]; labels=[]
    placeable=set().union(*(set(choices) for choices in catalog['choices'].values()))
    with Image.open(png) as sheet:
        for t in atlas['textures']:
            name=t['id'].split('/')[-1]
            if name not in placeable: continue
            x,y,w,h=t['rect']
            patch=sheet.crop((x,y,x+w,y+h)).resize((16,16),Image.Resampling.BOX)
            if patch.convert('RGBA').getchannel('A').getextrema()[0]<255: continue
            images.append(np.asarray(patch.convert('RGB')));labels.append(name)
    tiles=np.stack(images)
    return np.concatenate([tiles,degrade(tiles,10,(.8,.8,.8))]),np.tile(labels,2),'multiscale'


def reconstruct(image, atlas_dir, scene, library, out, depth=None, grid=None, white_balance=False):
    out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((atlas_dir/'placement-index.json').read_text())
    if grid:
        data=json.loads(grid.read_text())
        if data['image_sha256']!=digest(image): raise ValueError('Grid belongs to a different image')
        with Image.open(image) as f: rgb=np.asarray(ImageOps.exif_transpose(f).convert('RGB'))
        planes,cells=data['planes'],data['cells']
    else:
        rgb,planes,cells=analyze(image,scene,out)
    lighting=None
    if white_balance:
        im=Image.fromarray(rgb)
        gains,lighting=cast_estimate(im,[],[0,0,*im.size])
        if lighting['confidence']=='moderate': rgb=np.asarray(correct(im,gains))
    tiles,labels,method=exemplar_bank(atlas_dir,library,catalog)
    grouped=defaultdict(list)
    for cell in cells: grouped[cell['facing']].append(cell)
    blocks={}; margins={}; decisions=[]
    for side,part in grouped.items():
        mapping=catalog['choices'][side]
        use=np.isin(labels,list(mapping))
        train=features(tiles[use])
        for start in range(0,len(part),96):
            batch=part[start:start+96]
            query=np.stack([face(rgb,c['quad'],16) for c in batch])
            for c,(names,ds,gap) in zip(batch,rank(query,train,labels[use],method)):
                # Prefer a namesake full block among visually equivalent IDs.
                texture=names[0]; block=min(mapping[texture],key=lambda s:(s!=texture,len(s),s))
                pos=tuple(c['xyz'])
                if pos not in margins or gap>margins[pos]:
                    blocks[pos]=block;margins[pos]=gap
                decisions.append(dict(plane=c['plane'],ij=c['ij'],xyz=c['xyz'],block_id=block,
                                      textures=names,distances=ds,relative_margin=gap))
    if depth is not None:
        if len(planes)!=1 or planes[0]['facing'] in ROW_UP or type(depth)!=int or not 1<=depth<=256:
            raise ValueError('--depth needs a single vertical wall and 1 to 256 blocks')
        p=planes[0]; dx,_,dz=RIGHT[p['facing']]
        for c in cells:
            x,y,z=c['xyz']; i,j=c['ij']; block=blocks[x,y,z]
            for k in range(1,depth):
                if k==depth-1 or i in (0,p['columns']-1) or j in (0,p['rows']-1):
                    blocks[x+dz*k,y,z-dx*k]=block
    (out/'matches.json').write_text(json.dumps(dict(method=method,rendered_library=bool(library),
        geometry_input=grid.name if grid else 'geometry.json',image_sha256=digest(image),lighting=lighting,cells=decisions)))
    rows=[dict(x=x,y=y,z=z,block_id=name) for (x,y,z),name in sorted(blocks.items())]
    bounds=[[min(v),max(v)] for v in zip(*blocks)]
    (out/'placements.json').write_text(json.dumps(dict(version=catalog['version'],bounds_xyz=bounds,blocks=rows)))
    write_pack(out/'reconstruction.zip',blocks,catalog)
    print(f"{catalog['version']}: {len(planes)} planes, {len(blocks)} blocks -> {out/'reconstruction.zip'}")
