
"""Replay camera-free face assembly against frozen sky composites."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark.expanded_scores import load_reference
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import assemble, axis_vector, project
from benchmark.relative_face_scoring import evaluate_relative_voxels, yaw_from_anchor_normal
from engine.export import JAVA_26_3_FORMAT, write_pack


SHARED = SOURCE / 'floating-build-grid-01'
OUTPUT = ROOT / 'runs/relative-face-assembly-01'
ATLAS = WORKTREES / 'relative-face-source-charts/runs/relative-face-assembly-01'
CONTROL = WORKTREES / 'relative-face-junction-control/runs/relative-face-assembly-01'
MODES = {
    'atlas': (ATLAS,'charts.json',False,2,3.5,None,False),
    'atlas-one-edge': (ATLAS,'charts.json',False,1,3.5,None,False),
    'count-exploratory': (ATLAS,'counter-charts.json',False,1,3.5,.5,False),
    'count-depth-veto': (ATLAS,'counter-charts.json',False,1,3.5,.5,True),
    'exact-edges': (CONTROL,'charts.json',True,1,.01,None,False),
    'exact-edges-coplanar': (CONTROL,'charts.json',True,1,.01,.01,False),
    'atlas-dense': (ATLAS,'dense-charts.json',False,1,2.5,.5,False),
    'atlas-dense-depth': (ATLAS,'dense-charts.json',False,1,2.5,.5,True),
}


def load_depth(case, variant, size):
    path = SHARED/'baseline'/variant/case['dataset']/case['id']/'handoffs/depth.npz'
    with np.load(path,allow_pickle=False) as state:
        scalar = Image.fromarray(state['value'])
    return np.asarray(scalar.resize(size,Image.Resampling.BILINEAR)), digest(path)


def evidence_overlay(image, charts, result, selected, path):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    chosen = selected['origins']
    for chart in charts:
        h = np.asarray(chart['H'],float)
        for face in chart['cells']:
            x,y=face['u'],face['v']
            outline=[project(h,p) for p in [(x,y),(x+1,y),(x+1,y+1),(x,y+1)]]
            if any(c is None for c in outline):
                continue
            shape=[tuple(p) for p in outline]
            color = '#6df0af' if chart['id'] in chosen else '#8eb6ec'
            draw.line(shape+[shape[0]],fill=color,width=1)
    for join in result['joins']:
        if all(id in chosen for id in join['chart_ids']):
            for segment in join['image_edges']:
                draw.line([tuple(p) for p in segment],fill='#ffc963',width=3)
    image.save(path,quality=88,optimize=True)


def evaluate(job):
    case,variant,mode=job
    tree,name,reference_only,edges,edge_px,coplanar_px,depth=MODES[mode]
    input_file=tree/variant/case['id']/name
    if not input_file.is_file():
        return None
    packet=read(input_file)
    image_path=Path(case['variants'][variant]['image'])
    sha=digest(image_path)
    if sha!=packet['source_sha256'] or sha!=case['variants'][variant]['source_sha256']:
        raise ValueError(f'Chart and source images disagree: {input_file}')
    charts=packet['charts']
    image=Image.open(image_path).convert('RGB')
    field,field_sha=load_depth(case,variant,image.size) if depth else (None,None)
    result=assemble(charts,min_edges=edges,edge_pixels=edge_px,coplanar_pixels=coplanar_px,relative_depth=field)
    main=result['groups'][0] if result['groups'] else dict(blocks={},origins={})
    # Export is allowed at any anchor. Retain one connected group until the
    # source image establishes how detached groups relate to that same origin.
    predicted={p:'minecraft:'+material.removeprefix('minecraft:') for p,material in main['blocks'].items()}
    folder=OUTPUT/mode/variant/case['id']
    folder.mkdir(parents=True,exist_ok=True)
    pack=folder/'reconstruction.zip'
    write_pack(pack,{p:name.removeprefix('minecraft:') for p,name in predicted.items()},JAVA_26_3_FORMAT)
    if decoded_pack(pack)!=predicted:
        raise ValueError(f'Pack coordinates or IDs changed after export: {pack}')
    ref=load_reference(case)
    anchor=result['orientation_hint']
    yaw = None
    if anchor:
        yaw=yaw_from_anchor_normal(anchor['normal_camera'], axis_vector(anchor['local_normal_axis']), ref.pose)
    quarters=0 if reference_only or not predicted else (None if yaw is None else yaw['yaw_quarters'])
    score,aligned=evaluate_relative_voxels(predicted,ref,yaw_quarters=quarters,
                                            relaxed_yaw=not reference_only)
    evidence_overlay(image,charts,result,main,folder/'source-overlay.jpg')
    save(folder/'placements.json',dict(local=[[list(p),m] for p,m in sorted(predicted.items())],
        aligned=[[list(p),m] for p,m in sorted(aligned.items())]))
    receipt=dict(id=case['id'],dataset=case['dataset'], split=case['split'], background=variant, mode=mode,
                 reference_only=reference_only, source_sha256=sha, reference_sha256=ref.hashes,
                 chart_sha256=digest(input_file), pack_sha256=digest(pack), relative_depth_field_sha256=field_sha,
                 charts_seen=len(charts), oriented_charts=result['oriented_charts'], kept_charts=len(main['origins']),
                 components=len(result['groups']), component_sizes=[len(c['blocks']) for c in result['groups']],
                 selected_origins=main['origins'], joins=result['joins'], conflicts=result['conflicts'],
                 depth_vetoes=result['depth_vetoes'], yaw=yaw, score=score)
    save(folder/'score.json',receipt)
    display=score['fixed'] or score['relaxed_yaw']
    return (mode,variant,case['id'],len(charts),len(main['origins']),
            len(result['joins']),len(result['conflicts']),len(predicted),display['geometry']['strict'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modes',nargs='+',choices=MODES)
    parser.add_argument('--background',nargs='+',choices=('pavement','plain','sky'),default=['sky'])
    parser.add_argument('--workers',type=int,default=4)
    cfg=parser.parse_args()
    pairs=read(SHARED/'fixtures.json')['cases']
    jobs=[(case,variant,mode) for mode in cfg.modes for variant in cfg.background for case in pairs]
    with ProcessPoolExecutor(cfg.workers) as workers:
        for score in workers.map(evaluate,jobs):
            print(score,flush=True)


if __name__=='__main__':
    main()
