"""Audit in-game camera condition, requested block ID, lighting and framebuffer size."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from calibration.projection import camera_project,south_quad
from engine.nbt import read_nbt


def load(path):
    return json.loads(path.read_text())


def xyz(row):
    return tuple(row[k] for k in ('x','y','z'))


def saved_blocks(run):
    """Check the saved scene, including the negative space in the structural probes."""
    region=run/'client/saves/Palette Lab/dimensions/minecraft/overworld/region'
    @lru_cache(maxsize=None)
    def sections(cx,cz):
        path=region/f'r.{cx//32}.{cz//32}.mca'
        with path.open('rb') as f:
            f.seek(4*((cz%32)*32+cx%32))
            sector=int.from_bytes(f.read(3),'big')
            if not sector: raise ValueError(f'Missing chunk {cx},{cz}')
            f.seek(sector*4096)
            size=int.from_bytes(f.read(4),'big')
            if f.read(1)!=b'\x02': raise ValueError(f'Unsupported region compression in {path.name}')
            nbt=read_nbt(zlib.decompress(f.read(size-1)))
        if (nbt['xPos'],nbt['zPos'])!=(cx,cz): raise ValueError('Wrong saved chunk coordinates')
        return {s['Y']:s['block_states'] for s in nbt['sections'] if 'block_states' in s}
    def get(pos):
        x,y,z=pos
        sec=sections(x//16,z//16).get(y//16)
        if not sec: return 'air'
        palette=sec['palette'];idx=0
        if len(palette)>1:
            bits=max(4,(len(palette)-1).bit_length());per=64//bits
            k=(x%16)+16*(z%16)+256*(y%16)
            idx=(sec['data'][k//per]>>(bits*(k%per)))&((1<<bits)-1)
        state=palette[idx]
        name=state if isinstance(state,str) else state.get('id',state.get('Name',state.get('')))
        return name.removeprefix('minecraft:')
    cells=load(run/'layout.json')['blocks']
    bad=[xyz(b) for b in cells if get(xyz(b))!=b['block_id']]
    probes=load(run/'buildings.json');probe_counts=Counter();bad_probes=[]
    for b in probes:
        for p in b['probes']:
            pos=tuple(int(i) for i in np.floor(p['point']))
            if (get(pos)!='air')!=p['occupied']: bad_probes.append((b['id'],p['id']))
            probe_counts['solid' if p['occupied'] else 'air']+=1
    return dict(occupied_checked=len(cells),block_mismatches=bad,structural_probes=probe_counts,
                probe_mismatches=bad_probes)


def summarize(run,expected_pack=None):
    views=load(run/'views.json');layout=load(run/'layout.json')
    positions=defaultdict(dict)
    for b in layout['blocks']:
        positions[b['page']][xyz(b)]=b
    measured=[]; lights=Counter(); sizes=set(); ids=0
    conditions=set();by_case=defaultdict(set);failures=[]
    for v in views:
        stem=run/'captures'/v['id']
        p=load(stem.with_suffix('.json'))
        with Image.open(stem.with_suffix('.png')) as im:
            if im.size!=tuple(p['image_size']): raise ValueError('Frame dimensions differ from camera readback')
            sizes.add(im.size)
        if p['minecraft_version']!=layout['version'] or not p['hud_hidden'] or p['raining'] or p['thundering'] or p['clock_ticks']%24000!=6000:
            raise ValueError(f'Capture conditions failed in {v["id"]}')
        packs=p['resource_packs'];conditions.add(tuple(packs))
        if expected_pack and expected_pack not in packs: raise ValueError('Requested shader pack missing')
        actual={xyz(b):b for b in load(stem.with_name(stem.name+'-blocks.json'))}
        expected=positions[v['page']]
        if set(actual)!=set(expected): raise ValueError(f'Readback coordinates differ: {v["id"]}')
        for pos,b in expected.items():
            seen=actual[pos]
            if seen['block_id']!=b['block_id']: failures.append((v['id'],pos,b['block_id'],seen['block_id']))
            if b.get('role')=='adjacency_wall':
                q=camera_project(south_quad(*pos),p)
                edges=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
                measured.append(float(edges.min()))
                lights[str(seen['south_sky_light'])]+=1
        by_case[v['building_id']].add(v['angle']);ids+=len(expected)
    angles={'front','left','right','elevated'}
    if any(a!=angles for a in by_case.values()): raise ValueError('Building missing a camera angle')
    return dict(buildings=len(by_case),views=len(views),verified_coordinates=ids,mismatches=failures,
                resolutions=sorted(sizes),resource_packs=sorted(conditions),south_skylight_histogram=lights,
                adjacent_wall_short_edge_px_percentiles=dict(zip(('p10','p50','p90'),np.round(np.percentile(measured,[10,50,90]),2))) if measured else {})


def gallery(root):
    views=load(root/'views.json')
    families=list(dict.fromkeys(v['kind'] for v in views))
    angles=('front','left','right','elevated')
    canvas=Image.new('RGB',(4*320,len(families)*205),'#20262a')
    draw=ImageDraw.Draw(canvas)
    for j,f in enumerate(families):
        eligible=[v for v in views if v['kind']==f and v['split']=='test'] or [v for v in views if v['kind']==f]
        # Use one physical building for all four pictures in a row.
        building=eligible[len(eligible)//2]['building_id']
        for i,a in enumerate(angles):
            v=next(v for v in eligible if v['building_id']==building and v['angle']==a)
            with Image.open(root/'captures'/(v['id']+'.png')) as im:
                canvas.paste(ImageOps.contain(im.convert('RGB'),(316,179)),(i*320,j*205))
            draw.text((i*320+5,j*205+180),f.replace('_',' ')+' / '+a,fill='white')
    canvas.save(root/'benchmark-views.jpg',quality=90)


def shader_delta(clean,shader):
    views=load(shader/'views.json')
    shifts=[]
    for v in views:
        cleanstem=clean/'captures'/v['id']; altered=shader/'captures'/v['id']
        p=load(cleanstem.with_suffix('.json')); other=load(altered.with_suffix('.json'))
        for key in ('eye_world','yaw_pitch_deg','image_size','vertical_fov_deg'):
            if not np.allclose(p[key],other[key],atol=1.e-3): raise ValueError('Mismatched control camera')
        a=np.asarray(Image.open(cleanstem.with_suffix('.png')).convert('RGB'),dtype=np.int16)
        b=np.asarray(Image.open(altered.with_suffix('.png')).convert('RGB'),dtype=np.int16)
        coords=load(cleanstem.with_name(cleanstem.name+'-blocks.json'))
        points=[[t['x']+.5,t['y']+.5,t['z']+1] for t in coords if t['z']==0]
        q=camera_project(points,p).astype(int)
        q=q[(q[:,0]>=0)&(q[:,0]<a.shape[1])&(q[:,1]>=0)&(q[:,1]<a.shape[0])]
        shifts.extend(b[q[:,1],q[:,0]]-a[q[:,1],q[:,0]])
    shifts=np.array(shifts)
    return dict(paired_center_pixels=len(shifts),mean_absolute_channel_delta=round(float(abs(shifts).mean()),3),
                mean_rgb_delta=np.round(shifts.mean(0),3).tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--geometry-only',action='store_true',help='Audit a geometry-only suite without texture condition clients')
    args=p.parse_args()
    root=args.root;conditions=root/'conditions'
    runs=dict(clean=summarize(root))
    if not args.geometry_only:
        runs.update(small=summarize(conditions/'small'),warm_shader=summarize(conditions/'warm-shader','file/capture-profile.zip'))
    result=dict(runs)
    if not args.geometry_only:
        result['shader_framebuffer_change']=shader_delta(root,conditions/'warm-shader')
    result['saved_world']=saved_blocks(root)
    gallery(root)
    (root/'capture-audit.json').write_text(json.dumps(result,indent=2))
    if any(v['mismatches'] for v in runs.values()) or any(result['saved_world'][k] for k in ('block_mismatches','probe_mismatches')):
        raise SystemExit('World readback mismatches; see capture-audit.json')
