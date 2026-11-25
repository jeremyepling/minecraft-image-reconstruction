"""Capture flush facade controls with a dark inlay or a real skylight-shading overhang."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark.color_gallery import load
from calibration.world import create_world,PACK


def create(source,jar,out,building,positive_depths=None,rear_materials=None):
    source_building=next(b for b in load(source/'buildings.json') if b['id']==building)
    source_layout=load(source/'layout.json')
    digest=hashlib.sha256(jar.read_bytes()).hexdigest()
    if source_layout['client_sha256']!=digest:
        raise ValueError('Controls need the same client as the source fixture')
    with ZipFile(jar) as z:
        release=json.loads(z.read('version.json'))
        if rear_materials is not None:
            if positive_depths is None or len(rear_materials)!=len(positive_depths):
                raise ValueError('Supply one rear material for each requested opening depth')
            if any(f'assets/minecraft/blockstates/{m}.json' not in z.namelist() for m in rear_materials):
                raise ValueError('Rear material must name a vanilla block from the fixture version')
    source_views=[v for v in load(source/'views.json') if v['building_id']==building]
    lo,hi=source_building['bounds']
    dark={(x,y) for r in source_building['regions'] if r['kind']=='recess'
          for x in range(r['bounds'][0][0],r['bounds'][1][0])
          for y in range(r['bounds'][0][1],r['bounds'][1][1])}
    if not dark or not source_views:
        raise ValueError('A recessed facade with known openings and camera poses is required')
    material=next(b['block_id'] for b in source_layout['blocks']
                  if b['building_id']==building and b['role']=='facade')
    world=out/'fixture/saves/Palette Lab'
    if world.exists():raise FileExistsError(world)
    create_world(world,release,{'minecraft:random_tick_speed':0},level_name='Flat facade controls')
    dst=world/'datapacks'/PACK
    dst.parent.mkdir(parents=True,exist_ok=True)
    pv=release['pack_version']
    version=[pv['data_major'],pv['data_minor']]
    blocks=[];views=[];controls=[]
    with ZipFile(dst,'w',ZIP_DEFLATED) as pack:
        pack.writestr('pack.mcmeta',json.dumps({'pack':{'description':'Zero-depth facade tests',
                'min_format':version,'max_format':version}}))
        variants=('heldout-facade',) if positive_depths is not None else ('dark-inlay','physical-shade')
        for page,kind in enumerate(variants):
            dx=page*96
            cells={(x+dx,y,z):(material,'facade') for x in range(lo[0],hi[0])
                   for y in range(lo[1],hi[1]) for z in range(lo[2],hi[2])}
            if kind=='dark-inlay':
                for x,y in dark: cells[x+dx,y,hi[2]-1]=('polished_deepslate','flush_inlay')
            elif kind=='heldout-facade':
                openings=[r for r in source_building['regions'] if r['kind']=='recess']
                if len(openings)!=len(positive_depths) or any(d<0 or d>=hi[2]-lo[2] for d in positive_depths):
                    raise ValueError('Supply one depth per opening, retaining at least one back-wall block')
                for index,(r,d) in enumerate(zip(openings,positive_depths)):
                    low,high=r['bounds']
                    for x in range(low[0],high[0]):
                        for y in range(low[1],high[1]):
                            for z in range(hi[2]-d,hi[2]): del cells[x+dx,y,z]
                            if rear_materials is not None:
                                cells[x+dx,y,hi[2]-d-1]=(rear_materials[index],'recess_back' if d else 'flush_inlay')
            else:
                # A separate canopy above the facade casts daylight shade across
                # its left half. Every wall cell below still has identical depth.
                for x in range(lo[0],lo[0]+(hi[0]-lo[0])//2):
                    for z in range(hi[2]-1,hi[2]+5):
                        cells[x+dx,hi[1]+1,z]=(material,'shade_roof')
            commands=[]
            for (x,y,z),(block,role) in sorted(cells.items()):
                commands.append(f'setblock {x} {y} {z} minecraft:{block} strict')
                blocks.append(dict(x=x,y=y,z=z,block_id=block,state=block,role=role,page=page,
                                   kind=kind,building_id=kind,split='test'))
            pack.writestr(f'data/palette_lab/function/page_{page}.mcfunction','\n'.join(commands)+'\n')
            for v in source_views:
                target=list(v['target']);target[0]+=dx
                views.append(dict(v,id=kind+'-'+v['angle'],page=page,x=v['x']+dx,target=target,
                                  kind=kind,building_id=kind))
            regions=[dict(id=r['id'],kind='flush_control',bounds=[[r['bounds'][0][0]+dx,r['bounds'][0][1],hi[2]-1],
                       [r['bounds'][1][0]+dx,r['bounds'][1][1],hi[2]]],depth=0)
                     for r in source_building['regions'] if r['kind']=='recess']
            if positive_depths is not None:
                for region,depth in zip(regions,positive_depths):
                    region['kind']='heldout_opening'
                    region['depth']=depth
                    region['bounds'][0][2]=hi[2]-depth-1
                if rear_materials is not None:
                    for region,rear in zip(regions,rear_materials):
                        region['rear_material']=rear
                        low,high=region['bounds']
                        # Read back the empty reveal as well as its back surface.
                        for x in range(low[0],high[0]):
                            for y in range(low[1],high[1]):
                                for z in range(low[2]+1,high[2]):
                                    blocks.append(dict(x=x,y=y,z=z,block_id='air',state='air',role='recess_air',
                                                       page=page,kind=kind,building_id=kind,split='test'))
            controls.append(dict(id=kind,page=page,split='test',kind=kind,front_z=hi[2],actual_recess_depth=0,
                      facade_bounds=[[lo[0]+dx,lo[1],lo[2]],[hi[0]+dx,hi[1],hi[2]]],regions=regions))
            if positive_depths is not None:
                controls[-1].pop('actual_recess_depth')
    for name,data in [('layout',dict(source_layout,blocks=blocks)),('views',views),('buildings',controls)]:
        (out/(name+'.json')).write_text(json.dumps(data))


def audit(run):
    pages={b['page']:b for b in load(run/'buildings.json')}
    cells=load(run/'layout.json')['blocks']
    results={}
    for v in load(run/'views.json'):
        page=pages[v['page']]
        expected={(b['x'],b['y'],b['z']):b['block_id'] for b in cells if b['page']==v['page']}
        stem=run/'captures'/v['id']
        rows=load(stem.with_name(stem.name+'-blocks.json'))
        actual={(b['x'],b['y'],b['z']):b['block_id'] for b in rows}
        if actual!=expected:
            raise ValueError(f'Placed materials disagree in {v["id"]}')
        lo,hi=page['facade_bounds'];facade={}
        for b in rows:
            if lo[1]<=b['y']<hi[1] and b['block_id']!='air':
                key=(b['x'],b['y'])
                if key not in facade or b['z']>facade[key]['z']:
                    facade[key]=b
        front={(x,y):hi[2]-1 for x in range(lo[0],hi[0]) for y in range(lo[1],hi[1])}
        opening_depths={}
        opening_materials={}
        for region in page['regions']:
            low,high=region['bounds']
            sample={(x,y) for x in range(low[0],high[0]) for y in range(low[1],high[1])}
            for key in sample: front[key]-=region['depth']
            measured={hi[2]-1-facade[key]['z'] for key in sample}
            if measured!={region['depth']}:raise ValueError(f'Incorrect opening {region["id"]}')
            opening_depths[region['id']]=measured.pop()
            if 'rear_material' in region:
                observed={facade[key]['block_id'] for key in sample}
                if observed!={region['rear_material']}:raise ValueError(f'Incorrect material at {region["id"]}')
                opening_materials[region['id']]=observed.pop()
        if front!={pos:b['z'] for pos,b in facade.items()}:
            raise ValueError(f'Facade surface differs from control in {v["id"]}')
        light=[b['south_sky_light'] for b in facade.values()]
        if v['kind']=='physical-shade' and min(light)>=max(light):
            raise ValueError('Occluder produced no real skylight difference')
        results[v['id']]=dict(verified_blocks=len(actual),facade_cells=len(facade),measured_relief=max(opening_depths.values()),
                              opening_depths=opening_depths,
                              skylight_range=[min(light),max(light)])
        if opening_materials:results[v['id']]['opening_materials']=opening_materials
    (run/'control-audit.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','jar','out'):parser.add_argument('--'+name,type=Path,required=name=='out')
    parser.add_argument('--building',default='test-recess-28')
    parser.add_argument('--positive-depths',type=int,nargs='+',help='Build an independent relief pattern in original opening order')
    parser.add_argument('--rear-materials',nargs='+',help='Override only each opening back surface, including any flush inlay')
    parser.add_argument('--audit',action='store_true',help='Check the per-view visual controls after capture')
    args=parser.parse_args()
    if args.audit:
        audit(args.out)
    elif args.source and args.jar:
        create(args.source,args.jar,args.out,args.building,args.positive_depths,args.rear_materials)
    else:
        parser.error('--source and --jar are required to generate the controls')
