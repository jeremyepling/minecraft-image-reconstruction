"""Cache relative inverse depth from the installed MoGe-2 point head, one PNG at a time."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.normal_round import groups
from engine.surface_fields import MODELS, SETTINGS, session
from runtime.mode import image_only


def relative_depth(points, mask):
    """Estimate the unknown z offset using only the point map's pinhole projection."""
    h, w = mask.shape
    step = max(1, max(h, w) // 64)
    yy, xx = np.mgrid[step//2:h:step, step//2:w:step]
    ok = mask[yy,xx] > .5
    x, y, z = points[yy[ok], xx[ok]].astype(np.float64).T
    length = np.hypot(w,h)
    u = (2*xx[ok]+1-w)/length
    v = (2*yy[ok]+1-h)/length
    # f * X = u * (Z + shift), and likewise for Y. Weight later fits by projected error.
    design = np.stack((np.r_[x,y], -np.r_[u,v]), axis=1)
    target = np.r_[u*z,v*z]
    weight = np.ones_like(target)
    for _ in range(3):
        (focal, shift), _, rank, _ = np.linalg.lstsq(design*weight[:,None], target*weight, rcond=None)
        if rank < 2 or focal <= 0 or not np.isfinite(shift):
            raise ValueError('MoGe point field does not constrain a positive pinhole fit')
        dist = z+shift
        floor = max(1.e-6, float(np.percentile(dist, 5)))
        weight = np.tile(1/np.maximum(dist,floor),2)
    camera_z = points[...,2]+shift
    valid = (mask>.5) & (camera_z>0)
    result = np.zeros(mask.shape, np.float32)
    result[valid] = 1 / camera_z[valid]
    if not np.isfinite(result).all() or not np.any(valid):
        raise ValueError('Invalid MoGe relative-depth map')
    return result


@image_only()
def prepare(cache):
    provenance = dict(checkpoint=MODELS['normal'][2], preprocessing=SETTINGS,
                      adapter=digest(Path(__file__)), numpy=np.__version__)
    index_path = cache/'index.json'
    index = read(index_path) if index_path.is_file() else dict(recipe=provenance,fields={})
    if index['recipe'] != provenance:
        raise ValueError(f'Use a separate depth cache for this recipe: {cache}')
    cache.mkdir(parents=True,exist_ok=True)
    net = None
    for cases in groups()[0].values():
        for case in cases:
            sha = case.frame['image_sha256']
            path = cache/(sha+'.npy')
            if sha in index['fields']:
                if digest(path) != index['fields'][sha]['sha256']:
                    raise ValueError(f'Cached MoGe depth changed: {sha}')
                continue
            if digest(case.path('image')) != sha:
                raise ValueError(f'Screenshot changed: {case.id}')
            if net is None:
                env, filename, _ = MODELS['normal']
                sources = [ROOT/'.deps'/filename,ROOT.parent/'normal-round/.deps'/filename]
                if os.environ.get(env):
                    sources.insert(0,Path(os.environ[env]))
                checkpoint = next((p for p in sources if p.is_file()),sources[0])
                net = session('normal',checkpoint)
            with Image.open(case.path('image')) as opened:
                image = opened.convert('RGB')
            image.thumbnail((SETTINGS['normal_max_side'],)*2,Image.Resampling.LANCZOS)
            rgb = (np.asarray(image,np.float32)/255).transpose(2,0,1)[None]
            start = perf_counter()
            points, mask = net.run(['points','mask'],dict(image=rgb,num_tokens=np.array(SETTINGS['normal_tokens'],np.int64)))
            depth = relative_depth(points[0],mask[0])
            elapsed = perf_counter()-start
            with NamedTemporaryFile(dir=cache,delete=False) as tmp:
                temporary = Path(tmp.name)
                np.save(tmp,depth,allow_pickle=False)
            temporary.replace(path)
            index['fields'][sha] = dict(sha256=digest(path), inference_and_conversion_s=elapsed)
            save(index_path,index)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    print(len(prepare(parser.parse_args().cache)['fields']), 'unique relative-depth maps')
