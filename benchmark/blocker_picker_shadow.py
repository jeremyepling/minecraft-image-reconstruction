"""Replay Blocker Picker illumination and optional shape ranking on frozen raw face crops."""
from engine.face_evidence import MIN_FACE_WIDTH, effective_width
import argparse
import gzip
import json
from pathlib import Path
from collections import defaultdict
import importlib.util
import numpy as np
from PIL import Image
from engine.geometry import face
from engine.surface_model import canonical
from engine.blocker_shadow import surface_shadow_map, cell_shadow, corrected
from calibration.retrieval import rank,features
from benchmark.blocker_picker_exposure import sha

def run(cal, out, shape_module, plane_comparison=False):
    root=Path(__file__).resolve().parents[1]
    base=root/'benchmark/reviews'
    fixture=base/'mask-four-01'
    cases={c['key']:c for c in json.loads((fixture/'cases.json').read_text())}
    hashes=json.loads((fixture/'manifest.json').read_text())['artifact_sha256']
    materials=base/'count-01/materials.json'
    library=cal/'face_textures/softened-library.npz'
    idx=cal/'atlas/placement-index.json'
    index=json.loads(idx.read_text())
    mappings=defaultdict(set)
    for side in index['choices'].values():
        for tex,ids in side.items(): mappings[tex].update(ids)
    with np.load(library) as library_data:
        assert str(library_data['version'])=='26.3' and str(library_data['atlas_sha256'])==index['atlas_sha256']
        use=np.isin(library_data['labels'],list(mappings))
        train=features(library_data['tiles'][use]); labels=library_data['labels'][use]; method=str(library_data['method'])
    shape=None
    if shape_module:
        spec=importlib.util.spec_from_file_location('blocker_shape',shape_module)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        shape=module.ShapeMatcher(cal/'atlas')
    rows=[]; crops=[]
    out.parent.mkdir(parents=True,exist_ok=True)
    maps=out.parent/('plane-shadow-maps' if plane_comparison else 'shadow-maps');maps.mkdir(exist_ok=True)
    frames={}
    for frame in json.loads(materials.read_text())['pictures']:
        key=frame['key'];case=cases[key]
        image=fixture/case['source']
        masks=fixture/case['predictions']
        assert sha(image)==case['sha256'] and sha(masks)==hashes[case['predictions']]
        frames[key]=dict(image_sha256=sha(image),masks_sha256=sha(masks))
        rgb=np.asarray(Image.open(image).convert('RGB'))
        with gzip.open(masks) as src:
            field=surface_shadow_map(rgb,json.load(src)['mask-first'],plane_comparison)
        stem=key.replace('/','__')
        np.savez_compressed(maps/(stem+'.npz'),**field)
        preview=Image.fromarray(np.uint8(np.clip(255*(np.exp(field['loggain'])-1)/.40,0,255)))
        preview.resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.BILINEAR).save(maps/(stem+'.png'))
        for group in ('known','predicted'):
            for idx_row,item in enumerate(frame[group]):
                if 'truth' not in item or group=='known' and not(item['fixed_sample'] or item['paired_tile'] is not None):continue
                q=canonical(item['quad'])
                row=dict(key=key,index=idx_row,group=group,truth=item['truth'],quad=q,xyz=item['xyz'],fixed=item.get('fixed_sample',False),
                         paired=item.get('paired_tile') is not None,effective_width_px=effective_width(q))
                rows.append(row)
                if row['effective_width_px'] < MIN_FACE_WIDTH:
                    row.update(block=None,shape_block=None)
                    continue
                gain,details=cell_shadow(field,q,16)
                row['shadow']=details
                crops.append(corrected(face(rgb,q,16),gain))
                if shape:
                    g,_=cell_shadow(field,q,32)
                    result=shape.rank_patch(corrected(face(rgb,q,32),g))
                    row['shape_block']=result['block_id']
                    row['shape']=result
    eligible_rows=[r for r in rows if 'shadow' in r]
    rankings=rank(np.stack(crops),train,labels,method,True) if crops else ()
    for row,(names,d,gap) in zip(eligible_rows,rankings,strict=True):
        row.update(block=min(mappings[names[0]],key=lambda n:(n!=names[0],n)),textures=names,distances=d,margin=gap)
    score={}
    for group, selected in [('known_fixed',[r for r in rows if r['fixed']]),('known_paired',[r for r in rows if r['paired']]),('predicted',[r for r in rows if r['group']=='predicted'])]:
        total=len(selected)
        selected=[r for r in selected if r['effective_width_px']>=MIN_FACE_WIDTH]
        score[group]=dict(n=len(selected),total=total,under_minimum=total-len(selected),exposure_shadow=sum(r['truth']==r['block'] for r in selected))
        if shape: score[group]['shape_shadow']=sum(r['truth']==r['shape_block'].removeprefix('minecraft:') for r in selected)
    manifest=dict(min_face_width_px=MIN_FACE_WIDTH,plane_comparison=plane_comparison,scores=score,frames=frames,materials_sha256=sha(materials),index_sha256=sha(idx),atlas_sha256=index['atlas_sha256'],
                  library_sha256=sha(library),shadow_sha256=sha(root/'engine/blocker_shadow.py'),shape_sha256=sha(shape_module) if shape_module else None,
                  map_encoding='source / 12, white is >= 40% inferred light recovery, black is identity; NPZ has loggain, confidence, eave adjacency and surface ids',
                  rows=rows)
    out.write_text(json.dumps(manifest));print(score)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('calibration',type=Path);p.add_argument('out',type=Path);p.add_argument('--shape-module',type=Path)
    p.add_argument('--plane-comparison',action='store_true')
    args=p.parse_args();run(args.calibration,args.out,args.shape_module,args.plane_comparison)
