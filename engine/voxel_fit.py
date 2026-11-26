"""Generate a single-view 3D geometry sketch and an optional Minecraft structure pack."""
import json
import numpy as np
from PIL import Image, ImageFilter, ImageChops

from engine.export import write_pack
from engine.voxel_models import predict
from engine.voxel_space import prepare, render


def export(scene, grid, method, out, atlas=None):
    out.mkdir(parents=True,exist_ok=True)
    if not grid.any():
        raise ValueError('No occupied cells fit the image')
    points=np.argwhere(grid)
    model=dict(method=method,metric=bool(scene['metric']),grid_shape=list(grid.shape),
               projection=scene['P'].tolist(),projection_size=list(scene['mask'].shape[::-1]),
               occupied=points.tolist())
    if not model['metric']:
        model.update(units='relative sketch cells',sketch_resolution=int(scene['sketch_resolution']),
                     max_depth_fraction=float(scene['depth_fraction']))
    (out/'model.json').write_text(json.dumps(model))
    size=tuple(scene['mask'].shape[::-1])
    projected=render(grid,scene['P'],size,True)
    projected.save(out/'model.png')
    boundary=ImageChops.subtract(projected.filter(ImageFilter.MaxFilter(3)).convert('L'),
                                 projected.filter(ImageFilter.MinFilter(3)).convert('L')).point(lambda v:255 if v>8 else 0)
    overlay=Image.fromarray(scene['image']).copy()
    overlay.paste((255,95,42),(0,0,*size),boundary)
    overlay.save(out/'outline.png')
    if atlas:
        meta=json.loads((atlas/'placement-index.json').read_text())
        points[:,2] *= -1
        points -= points.min(0)
        blocks={tuple(map(int,p)):'smooth_stone' for p in points}
        write_pack(out/'reconstruction.zip',blocks,meta)
        placed=[dict(x=x,y=y,z=z,block_id=block) for (x,y,z),block in blocks.items()]
        (out/'placements.json').write_text(json.dumps(dict(version=meta['version'],blocks=placed)))


def fit(image, method, out, atlas=None, relative=False):
    scene=prepare(Image.open(image),allow_relative=relative)
    export(scene,predict(scene,method),method,out,atlas)
