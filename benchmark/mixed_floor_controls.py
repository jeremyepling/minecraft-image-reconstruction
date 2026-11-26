"""Repaint visible ground in three development screenshots for paired camera controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark.camera_metrics import registration, score, truth_camera
from benchmark.texture_camera_eval import candidate_fn, checked, frames, plain
from calibration.world import GROUND
from engine.atlas_count import prepared_atlas_paths
from engine.camera_grid import plane_coordinates
from engine.camera_seam import predict as production
from runtime.mode import mark_image_only


PALETTES = {
    'distinct': ('oak_planks', 'stone', 'bricks', 'prismarine_bricks',
                 'polished_diorite', 'dark_oak_planks', 'red_sandstone', 'obsidian'),
    'similar': ('stone','andesite','polished_andesite','stone_bricks',
                'cobblestone','smooth_stone','tuff','gravel'),
}


def sprites(names):
    atlas = prepared_atlas_paths('26.3')[0]
    meta = json.loads((atlas/'atlas.json').read_bytes())
    by_name = {t['id'].split('/')[-1]: t['rect'] for t in meta['textures']}
    with Image.open(atlas/meta['atlas']) as sheet:
        result = []
        for name in names:
            x, y, w, h = by_name[name]
            result.append(np.asarray(sheet.crop((x,y,x+w,y+h)).convert('RGB').resize((16,16),Image.Resampling.BOX)))
    return np.asarray(result)


def tile_indices(a,b):
    return 4*((a+b)%2) + ((a*1973 ^ b*9277 ^ a*b*283) % 4)


def make_image(image, pose, faces, tiles, native_sampling=False):
    w,h = image.size
    top = int(.63*h)
    keep = Image.new('L',(w,h))
    draw = ImageDraw.Draw(keep)
    for face in faces:
        draw.polygon([tuple(p) for p in face['image_corners']], fill=255)
    keep = np.asarray(keep.filter(ImageFilter.MaxFilter(3)), dtype=bool)[top:]
    py,px = np.mgrid[top:h, :w]
    P = truth_camera(pose)
    accum = np.zeros((*px.shape,3), np.float32)
    offsets = (.5,) if native_sampling else (.25,.75)
    for dy in offsets:
        for dx in offsets:
            xz = plane_coordinates(P,np.stack((px+dx,py+dy),-1).reshape(-1,2),1,GROUND+1).reshape(*px.shape,2)
            cell = np.floor(xz).astype(np.int64)
            a,b = cell.transpose(2,0,1)
            # Disjoint sprite sets on each parity guarantee that every shared
            # block edge changes texture. The order within each set varies.
            idx = tile_indices(a,b) if len(tiles)>1 else 0
            u,v = np.moveaxis(np.floor((xz-cell)*16).astype(int),-1,0)
            accum += tiles[idx,v,u]
    rgb = np.array(image)
    rgb[top:][~keep] = (accum[~keep]/len(offsets)**2).round().astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path)
    p.add_argument('--candidate')
    p.add_argument('--cohort',choices=['development','check'],default='development')
    p.add_argument('--angle',choices=['high-right','low-left'],default='high-right')
    p.add_argument('--uniform',action='store_true',help='Use one vanilla sprite over the visible floor.')
    p.add_argument('--native-sampling',action='store_true',help='Sample one texel per pixel center.')
    args = p.parse_args()
    mark_image_only()
    fn = candidate_fn(args.candidate_root.resolve(),args.candidate) if args.candidate else None
    root = args.data/'train'/'frames'
    rows = frames(args.data.parent,'train')
    families = sorted({r['family'] for r in rows})
    start = int(args.cohort=='check')
    ids = [min(r['id'] for r in rows if r['family']==fam and r['angle']==args.angle) for fam in families[start::2]]
    args.out.mkdir(parents=True,exist_ok=True)
    results = []
    palettes = {name:(name,) for name in ('light_gray_concrete','red_concrete','white_concrete',
                                          'oak_planks','stone')} if args.uniform else PALETTES
    banks = {k:sprites(v) for k,v in palettes.items()}
    for name in ids:
        f = root/name
        hashes = next(r['files'] for r in rows if r['id']==name)
        for file in ('rgb.png','camera.json','reference-faces.json'):
            checked(f/file,hashes[file])
        pose = json.loads((f/'camera.json').read_bytes())
        faces = json.loads((f/'reference-faces.json').read_bytes())['faces']
        with Image.open(f/'rgb.png') as opened:
            image = opened.convert('RGB')
        for palette in ('original',*banks):
            photo = image if palette=='original' else make_image(image,pose,faces,banks[palette],args.native_sampling)
            if palette!='original':
                photo.save(args.out/f'{name}-{palette}.png')
            pred = production(photo)
            used = fn is not None and pred['P'] is None
            if used:
                pred = fn(photo,'26.3')
            metrics = score({**pred,'planes':[]},pose,faces)
            edges = []
            if palette in banks and pred['P'] is not None:
                T = registration(np.asarray(pred['P']),pose)
                for edge in pred.get('evidence',{}).get('confirmed_edges',[]):
                    if 'first' not in edge:
                        continue
                    truth = []
                    for key in ('first','second'):
                        a,b = edge[key]
                        loc = np.floor(T @ [a+.5,0,b+.5,1]).astype(int)
                        names = palettes[palette]
                        truth.append(names[tile_indices(loc[0],loc[2])] if len(names)>1 else names[0])
                    edges.append(dict(first=edge['first'],second=edge['second'],
                                      predicted=[edge['a'],edge['b']],truth=truth))
            results.append(dict(id=name,palette=palette,candidate_invoked=used,
                                metric_camera=pred['P'] is not None,P=pred['P'],
                                reason=pred['reason'],atlas_edges=edges,**metrics))
    (args.out/'control.json').write_text(json.dumps(dict(palettes=palettes,candidate=args.candidate,
        native_sampling=args.native_sampling,frames=ids,results=results),default=plain))
    print([(r['palette'],r['within_2px'],r['vertices']) for r in results])


if __name__ == '__main__':
    main()
