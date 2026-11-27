"""Extract independent, source-supported unit grids without fitting a 3D camera."""
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from benchmark.floating_build_local_grid import planar_charts
from benchmark.full_block_lattice import atlas, rectified_sprite_hits, verify_sprite
from engine.geometry import project, transform, UNIT
from engine.visible_raster import observed_support
import cv2
import numpy as np
from PIL import Image


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def wall_quad(row, part):
    if not row.get('quad') or not part.get('normal_camera') or abs(part['normal_camera'][1]) > .63:
        return None
    q = np.asarray(row['quad'])
    across = q[1]-q[0]+q[2]-q[3]
    down = q[3]-q[0]+q[2]-q[1]
    if across[0] > abs(across[1]) and down[1] > abs(down[0]):
        return q


def inside(quad, mask):
    pixels = quad / 4 - .5
    lo = np.floor(pixels.min(0)).astype(int)
    hi = np.ceil(pixels.max(0)).astype(int) + 1
    if np.any(lo < 0) or np.any(hi > mask.shape[::-1]) or np.any(hi <= lo):
        return False
    footprint = np.zeros(tuple((hi-lo)[::-1]), 'B')
    cv2.fillConvexPoly(footprint, np.rint(pixels-lo).astype('i'), 1)
    return mask[lo[1]:hi[1],lo[0]:hi[0]][footprint.astype(bool)].all()


def source_cell(rgb, h, mask, u, v):
    q = project(h, UNIT + (u, v))
    if not np.isfinite(q).all() or np.linalg.norm(q-np.roll(q, 1, axis=0),axis=1).min() < 7:
        return
    if inside(q, mask):
        return q


def grow(rgb, h, mask):
    todo = deque([(0,0)])
    tried = set()
    accepted = []
    while todo:
        u,v = todo.popleft()
        if (u,v) in tried:
            continue
        tried.add((u,v))
        q = source_cell(rgb,h,mask,u,v)
        if q is None:
            continue
        hit = verify_sprite(rgb,q[[3,2,1,0]])
        if hit and hit[2] >= .68:
            accepted.append(dict(u=u,v=v,block_id=hit[0],confidence=hit[2]))
            todo.extend(((u+1,v),(u-1,v),(u,v+1),(u,v-1)))
    return accepted


def scaled_charts(rgb, q, mask, aspect):
    # Rectification needs only the shape of the observed surface. Absolute block size
    # and phase come from matching native atlas tiles inside its image mask.
    side = np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    ratio = aspect or (side[0]+side[2])/(side[1]+side[3])
    scale = min(132., max(side)) / max(ratio,1)
    w,h = np.rint([scale*ratio,scale]).astype(int)
    if min(w,h)<9:
        return
    warp = transform(q) @ np.diag([1/w,1/h,1])
    x,y = np.meshgrid(np.arange(w)+.5,np.arange(h)+.5)
    xy = project(warp,np.stack((x,y),axis=-1)).astype('f')
    crop = cv2.remap(rgb,xy[:,:,0]-.5,xy[:,:,1]-.5,cv2.INTER_LINEAR)
    valid = cv2.remap(mask.astype('B'),xy[:,:,0]/4-.5,xy[:,:,1]/4-.5,cv2.INTER_NEAREST)
    for seed in rectified_sprite_hits(crop,valid,warp,rgb):
        h_up = transform(np.array(seed['quad'])[[3,2,1,0]])
        cells = grow(rgb,h_up,mask)
        if len(cells)>=2 or (cells and cells[0]['confidence']>=.83):
            yield h_up,cells


def counter_chart(rgb,q,mask,row):
    counts = row.get('counts',[])
    if len(counts)!=2 or any(v is None or not 1<=v<=60 for v in counts):
        return
    # Missing phase uses the sampling-chart boundary as a deliberately weak hypothesis.
    phase = [v if v is not None else 0 for v in row.get('phase',[None,None])]
    w,h = counts
    warp = transform(q) @ np.array([[1/w,0,phase[0]/w],[0,-1/h,(phase[1]+np.floor(h-phase[1]))/h],[0,0,1]])
    cells=[]
    for u in range(int(w-phase[0])):
        for v in range(int(h-phase[1])):
            quad = source_cell(rgb,warp,mask,u,v)
            if quad is not None:
                hit = verify_sprite(rgb,quad[[3,2,1,0]])
                cells.append(dict(u=u,v=v,block_id=hit[0] if hit else 'smooth_stone',
                                  confidence=hit[2] if hit else 0.,fallback=hit is None))
    if cells:
        return warp,cells


def extract(task):
    spec, name, variant, handoff, out = task
    source=Path(spec['image'])
    if digest(source)!=spec['source_sha256']:
        raise ValueError('Unexpected source hash')
    def read(filename):
        return json.loads((handoff/filename).read_bytes())
    if read('build-manifest.json')['image_sha256']!=spec['source_sha256']:
        raise ValueError('Cached source mismatch')
    rows={r['id']:r for r in read('counts.json')['surfaces']}
    image=Image.open(source).convert('RGB')
    rgb=np.asarray(image)
    charts=[]
    exploratory=[]
    from benchmark.relative_dense_charts import dense_charts, vertical_pencil
    from benchmark.camera_structure_scaffold import outline_lines
    parts=read('surfaces.json')['planes']
    walls=[p for p in parts if wall_quad(rows.get(p['id'],{}),p) is not None]
    lines,_=outline_lines(image,walls,[])
    vert=vertical_pencil(lines,image.size)
    dense=[]
    for part in parts:
        row=rows.get(part['id'],{})
        q=wall_quad(row,part)
        if q is None or not part.get('support'):
            continue
        mask=observed_support(part['support'],(image.height//4,image.width//4),image.size)
        contours,_=cv2.findContours(mask.astype('B'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=[((cv2.approxPolyDP(c,1,True)[:,0,:]+.5)*4).tolist() for c in contours if len(c)>=3]
        common=dict(id=part['id'],normal_camera=part['normal_camera'],
                    relative_inverse_depth=part.get('relative_inverse_depth'),support_contours=contours,
                    surface_quad=q.tolist())
        proposals=[]
        for anchors,H in planar_charts(rgb,part,row,image.size):
            cells=[dict(u=a['cell'][0],v=a['cell'][1],block_id=a['block'],confidence=a['corr'])
                   for a in anchors]
            proposals.append(('counter-scale-atlas',H,cells))
        for H,cells in scaled_charts(rgb,q,mask,row.get('evidence',{}).get('aspect')):
            proposals.append(('multiscale-atlas',H,cells))
        if proposals:
            method,H,cells=max(proposals,key=lambda p:(len(p[2]),np.mean([c['confidence'] for c in p[2]])))
            charts.append(dict(common,method=method,H=H.tolist(),cells=cells,
                               confidence=float(np.mean([c['confidence'] for c in cells]))))
        for H,cells in dense_charts(rgb,q,mask,part['normal_camera'],row.get('evidence',{}).get('aspect'),vert):
            proposals.append(('local-atlas-window',H,cells))
        if proposals:
            method,H,cells=max(proposals,key=lambda p:(len(p[2]),np.mean([c['confidence'] for c in p[2]])))
            dense.append(dict(common,method=method,H=H.tolist(),cells=cells,
                              confidence=float(np.mean([c['confidence'] for c in cells]))))
        candidate=counter_chart(rgb,q,mask,row)
        if candidate is not None:
            H,cells=candidate
            exploratory.append(dict(common,method='counter-boundary-phase',H=H.tolist(),cells=cells,
                                    confidence=float(np.mean([c['confidence'] for c in cells]))))
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(id=name,variant=variant,source_sha256=spec['source_sha256'],
                  surfaces_sha256=digest(handoff/'surfaces.json'),counts_sha256=digest(handoff/'counts.json'),
                  atlas_sha256=atlas().provenance['atlas_sha256'],
                  units='Integer lower-left cell corner (u,v), u right and v up. H maps corners to source pixels.',
                  normal_convention='camera x right, y down, z forward')
    for filename,data in [('charts.json',charts),('counter-charts.json',exploratory),('dense-charts.json',dense)]:
        tmp=out/(filename+'.tmp')
        tmp.write_text(json.dumps(dict(manifest,charts=data),separators=(',',':')))
        tmp.replace(out/filename)
    print(variant,name,len(dense),sum(len(c['cells']) for c in dense),flush=True)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('handoffs',type=Path)
    parser.add_argument('--out',type=Path,default=Path('runs/relative-face-assembly-01'))
    parser.add_argument('--variants',nargs='+',default=['sky'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    tasks=[(e['variants'][v],e['id'],v,args.handoffs/v/e['id']/'source-handoff',args.out/v/e['id'])
           for v in args.variants for e in json.loads(args.manifest.read_bytes())['cases']]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(extract,tasks))


if __name__=='__main__':
    main()
