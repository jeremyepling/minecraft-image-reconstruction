"""Compare source, old, shared-palette and motif-palette live frames using frozen geometry."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmark.color_gallery import load
from calibration.projection import camera_project, south_quad
from engine.geometry import face

OPTIONS = ('old', 'shared', 'motif')
CORNERS = np.array(list(product((0, 1), repeat=3)))


def hull_half(points):
    hull = []
    for x, y in points:
        while len(hull) >= 2:
            (ax, ay), (bx, by) = hull[-2:]
            if (bx-ax)*(y-ay) > (by-ay)*(x-ax):
                break
            hull.pop()
        hull.append((x, y))
    return hull


def frame(captures, view, blocks):
    pose = load(captures/(view+'.json'))
    with Image.open(captures/(view+'.png')) as image:
        image = image.convert('RGB')
    if image.size != tuple(pose['image_size']):
        raise ValueError(f'Unexpected framebuffer size for {view}')
    mask = Image.new('L', image.size, 0)
    brush = ImageDraw.Draw(mask)
    for b in blocks:
        pos = np.array([b[k] for k in ('x','y','z')])
        points = sorted(set(map(tuple, camera_project(pos+CORNERS, pose))))
        brush.polygon(hull_half(points)[:-1]+hull_half(reversed(points))[:-1], fill=255)
    return image, mask, pose


def center_lab(rgb, quad):
    """D65 CIELAB of the median sRGB in the central 10x10 rectified pixels."""
    patch = face(rgb, np.asarray(quad), 16)[3:13, 3:13]
    c = np.median(patch.reshape(-1,3), axis=0)/255
    linear = np.where(c<=.04045, c/12.92, ((c+.055)/1.055)**2.4)
    xyz = np.array([[.4124564,.3575761,.1804375],
                    [.2126729,.7151522,.0721750],
                    [.0193339,.1191920,.9503041]]) @ linear
    rel = xyz/np.array([.95047,1,1.08883])
    x,y,z = np.where(rel>(6/29)**3,np.cbrt(rel),rel/(3*(6/29)**2)+4/29)
    return np.array([116*y-16,500*(x-y),200*(y-z)])


def targets(cells, image, mask):
    rgb = np.asarray(image)
    silhouette = np.repeat(np.asarray(mask)[:,:,None],3,axis=2)
    grouped = defaultdict(list)
    for cell in cells:
        if cell['facing']!='south':
            raise ValueError('Comparison requires visible south-facing preview cells')
        quad = np.array(cell['quad'])
        coverage = face(silhouette, quad,16)[3:13,3:13,0].mean()/255
        grouped[tuple(cell['xyz'])].append((coverage>.8, center_lab(rgb,quad)))
    result = {}
    for coord, samples in grouped.items():
        # Overlapping image planes may project onto the same output block.
        # Weight each physical coordinate once, regardless of the winning material.
        foreground = [color for inside,color in samples if inside]
        result[coord] = (np.median([c for _,c in samples],axis=0),
                         np.median(foreground,axis=0) if foreground else None)
    return result


def error_summary(diffs):
    if not diffs:
        return {'n':0}
    d = np.stack(diffs)
    distance = np.linalg.norm(d,axis=1)
    return dict(n=len(d),mean_delta_e76=float(distance.mean()),
                median_delta_e76=float(np.median(distance)),
                mean_abs_L=float(np.abs(d[:,0]).mean()),
                mean_chroma_error=float(np.linalg.norm(d[:,1:],axis=1).mean()))


def case(view, fixture, roots, fixture_blocks):
    view_id = view['id']
    source = frame(fixture/'captures',view_id,fixture_blocks)
    cells = load(roots['old']/view_id/'final-build/geometry.json')['cells']
    geometry_key = [(c['plane'],c['ij'],c['xyz']) for c in cells]
    samples = targets(cells,*source[:2])
    panels = [source]; colors = {}; readbacks = {}
    camera_keys = ('eye_world','yaw_pitch_deg','vertical_fov_deg','image_size')
    for name, directory in roots.items():
        build_name, run_name = ('final-build','final-verify') if name=='old' else ('build','verify')
        build,run = directory/view_id/build_name,directory/view_id/run_name
        matches = load(build/'matches.json')['cells']
        if geometry_key!=[(c['plane'],c['ij'],c['xyz']) for c in matches]:
            raise ValueError(f'Geometry differs between options in {view_id}')
        verified = load(run/'verification.json')
        if verified['differences'] or verified['expected']!=verified['matched']:
            raise ValueError(f'Live readback failed in {run}')
        placed = load(run/'layout.json')['blocks']
        offset = min(b['y'] for b in placed)-min(c[1] for c in samples)
        observed = {(b['x'],b['y']-offset,b['z']) for b in placed}
        if observed!=samples.keys():
            raise ValueError(f'Sample coordinates disagree with placement in {view_id}')
        im,mask,pose = frame(run/'captures','reconstruction',placed)
        if len(panels)>1 and any(not np.allclose(pose[k],panels[1][2][k],atol=.0001) for k in camera_keys):
            raise ValueError(f'Rebuild cameras differ in {view_id}')
        panels.append((im,mask,pose))
        rgb = np.asarray(im)
        whole, foreground = [], []
        for (x,y,z),(all_target,fg_target) in samples.items():
            quad = camera_project(south_quad(x,y+offset,z),pose)
            color = center_lab(rgb,quad)
            whole.append(color-all_target)
            if fg_target is not None:
                foreground.append(color-fg_target)
        readbacks[name]=verified['matched']
        colors[name]={'foreground':error_summary(foreground),'all':error_summary(whole)}
    return panels,dict(live_readbacks=readbacks,color=colors)


def thumbnail(image, mask):
    x,y,x1,y1 = mask.getbbox()
    pad = max(4,round(max(x1-x,y1-y)*.08))
    box = max(0,x-pad),max(0,y-pad),min(image.width,x1+pad),min(image.height,y1+pad)
    return ImageOps.contain(image.crop(box),(370,193),Image.Resampling.LANCZOS)


def label_font(size):
    for family in ('DejaVuSans.ttf','segoeui.ttf','Arial.ttf'):
        try:
            return ImageFont.truetype(family,size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0
        return ImageFont.load_default()


def compare(args):
    roots = {name:getattr(args,name) for name in OPTIONS}
    views = {v['id']:v for v in load(args.fixture/'views.json')}
    chosen = args.views or [v for v in views if all((p/v).is_dir() for p in roots.values())]
    if not chosen:
        raise ValueError('No common source and verified builds')
    pages = defaultdict(list)
    for b in load(args.fixture/'layout.json')['blocks']:
        pages[b['page']].append(b)
    canvas = Image.new('RGB',(1520,52+232*len(chosen)),'#1f2428')
    draw = ImageDraw.Draw(canvas); title = label_font(20); label = label_font(16)
    for col,name in enumerate(('Source','Old palette','Shared palette','Motif palette')):
        draw.text((col*380+6,9),name,font=title,fill='white')
    report = {}
    for row,view_id in enumerate(chosen):
        view = views[view_id]
        panels,report[view_id] = case(view,args.fixture,roots,pages[view['page']])
        for col,(image,mask,_) in enumerate(panels):
            thumb = thumbnail(image,mask)
            x,y = col*380,46+row*232
            canvas.paste(thumb,(x+(380-thumb.width)//2,y+(193-thumb.height)//2))
            if col==0:
                draw.text((x+6,y+197),view['kind'].replace('_',' ').capitalize(),font=label,fill='#d3d9dd')
    for path in (args.image,args.metrics):
        path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.image,quality=92,subsampling=0)
    args.metrics.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture',*OPTIONS,'image','metrics'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--views',nargs='+',help='Optional list of source view IDs in display order')
    compare(parser.parse_args())
