"""Generate independent texture, crop, orientation and occlusion controls."""
import argparse

import numpy as np
from PIL import Image

from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import STABLE, save
from engine.camera_atlas_floor import _atlas
from engine.front_wall_grid import local_grid


def controls(out):
    path = STABLE/'runs/26.3/atlas'
    atlas = _atlas('26.3',path)
    results = []
    for name,pitch in [('bricks',6.8),('oak_planks',8.2),('stone',5.7),('white_concrete',7.4)]:
        tile = atlas.tiles[atlas.labels.index(name)]
        w,h = 910,645
        x = np.floor((np.arange(w)+.5)/pitch+3.7).astype(int)
        y = np.floor((np.arange(h)+.5)/pitch+9.2).astype(int)
        hx,hy = x//16,y//16
        for occluded in (False,True):
            rgb = np.uint8(.79*tile[y[:,None]%16,x%16])
            visible = np.ones((h,w),bool)
            if occluded:
                visible[(hy[:,None]==2) & (hx==2)] = False
                rgb[~visible] = (41,153,97)
            im = Image.fromarray(rgb)
            cells,evidence = local_grid(im,atlas_path=path)
            screen_cells = {(int(xx),int(-yy)) for xx in np.unique(hx) for yy in np.unique(hy)
                            if np.any(visible[np.ix_(hy==yy,hx==xx)])}
            for orientation in ('wall','floor'):
                target = {(xx,yy,0) if orientation=='wall' else (xx,0,yy) for xx,yy in screen_cells}
                _,correct = best_translation(cells,target)
                results.append(dict(sprite=name,texel_pitch=pitch,occluded=occluded,orientation=orientation,
                                    target=len(target),placed=len(cells),correct=correct,extras=len(cells)-correct))
    save(out/'controls.json',results)
    for orientation in ('wall','floor'):
        rows = [r for r in results if r['orientation']==orientation]
        print(orientation,{k:sum(r[k] for r in rows) for k in ('target','correct','extras')})


if __name__=='__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    controls(args.out)

