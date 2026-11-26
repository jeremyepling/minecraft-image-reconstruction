"""Trace the production footprint gates on family surfaces with both grid offsets."""
import argparse
from pathlib import Path

import numpy as np

from benchmark.camera_round import digest, save
from benchmark.count_materials import true_faces
from benchmark.count_metrics import reference_planes
from benchmark.grid_placement_check import corner_error, CORNER_TOLERANCE
from benchmark.known_spacing_atlas import predict
from benchmark.mask_metrics import measure
from benchmark.phase_material_families import iter_family
from benchmark.textures import match_cells
from engine.geometry import UNIT, project, transform
from engine.masked_count import count
from engine.surface_sample import sample_surface
from engine.surface_tiles import complete_footprints


def trace(sample,grid):
    mask = sample['mask']
    height,width = mask.shape
    size = np.array([width,height])
    ns = np.asarray(grid['counts'])
    phase = np.mod(grid['phase'],1.)
    # Use the production chart and integer sampling bounds; record which gate rejects each cell.
    origins = [np.arange(int(np.floor(n-p+1e-7)))+p for n,p in zip(ns,phase)]
    cells = []
    inside_image = 0
    H = transform(sample['quad'])
    for y in origins[1]:
        for x in origins[0]:
            uv = (UNIT+[x,y])/ns
            lo = np.floor(uv[0]*size+1e-7).astype(int).clip([0,0],size)
            hi = np.ceil(uv[2]*size-1e-7).astype(int).clip([0,0],size)
            patch = mask[lo[1]:hi[1],lo[0]:hi[0]]
            quad = project(H,uv)
            cells.append(dict(supported_fraction=float(patch.mean()) if patch.size else 0.,quad=quad.tolist()))
            if patch.size and patch.all():
                inside_image += int(np.all(quad>=0) and np.all(quad<=sample['source_image'].size))
    actual = len(list(complete_footprints(sample,grid)))
    if actual!=inside_image:
        raise ValueError('Trace diverged from ordinary cell extraction')
    fractions = [c['supported_fraction'] for c in cells]
    return dict(counts=grid['counts'],phase=grid['phase'],sample_pixels=size.tolist(),diagnostic_cells=cells,
                complete_cells_across_down=[len(v) for v in origins],inside_chart=len(fractions),
                fully_supported=sum(f==1 for f in fractions),emitted=actual,
                best_supported_fraction=max(fractions,default=None))


def run(dataset, review, out):
    rows = []
    shards = {}
    for layout,split,archive in [('cloister1','validation',review/'cloister-surfaces.zip'),
                                ('twinspire1','test',review.parent/'block-seam-01/family/surfaces.zip')]:
        shard = dataset/f'visible-builds-v1-{split}.zip'
        shards[layout] = digest(shard)
        for frame,sha,image,parts,load_reference in iter_family(layout,shard,archive):
            traced = []
            for i,part in enumerate(parts):
                sample = sample_surface(image,part)
                grid = predict(sample,count(sample))
                if sample is not None and None not in grid['phase']:
                    traced.append((i,dict(frame=frame,sha256=sha,proposal=part['id'],**trace(sample,grid))))
            ref,faces = load_reference()
            valid = true_faces(reference_planes(ref,faces),image.size)
            paired = dict(measure(parts,ref,use_mask=True)[1])
            for i,row in traced:
                rid = paired.get(i)
                cells = row['diagnostic_cells']
                for c in cells:
                    c['aligned'] = False
                if rid is not None:
                    inv = np.linalg.inv(ref['charts'][rid])
                    by_quad = {id(c['quad']):c for c in cells}
                    targets = {uv for pid,uv in valid if pid==rid}
                    matched,_ = match_cells([c['quad'] for c in cells],targets,inv)
                    for uv,(quad,_,_) in matched.items():
                        by_quad[id(quad)]['aligned'] = corner_error(quad,uv,inv)<=CORNER_TOLERANCE
                rows.append(row)
    save(out,dict(shards=shards,surfaces=rows))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('review',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    run(args.dataset,args.review,args.out)
