"""Stage two additional, matched camera headings for previously generated facade fixtures."""
import argparse
import json
import math
from pathlib import Path
import shutil

from benchmark.color_gallery import load


def stage(source, out, buildings, angles=(-26,26)):
    if not -90 < angles[0] < 0 < angles[1] < 90:
        raise ValueError('Provide one negative left and one positive right facade heading between -90 and 90 degrees')
    manifest = load(source/'layout.json')
    original_views = load(source/'views.json')
    parents = [v for v in original_views if v['angle']=='left' and v['building_id'] in buildings]
    if {v['building_id'] for v in parents} != set(buildings):
        raise ValueError('Did not find one original left camera for each selected facade')
    if out.exists():
        raise FileExistsError(f'Stage into a fresh run directory: {out}')
    views = []
    for old in parents:
        tx,ty,tz = old['target']
        radius = math.hypot(old['x']-tx,old['z']-tz)
        # A matched elevation and radial distance isolate horizontal heading.
        for name,degrees in zip(('oblique-left','oblique-right'),angles):
            angle=math.radians(degrees)
            views.append(dict(old,id=old['building_id']+'-'+name,angle=name,
                x=tx+radius*math.sin(angle),z=tz+radius*math.cos(angle)))
    pages={v['page'] for v in views}
    controls=[b for b in load(source/'buildings.json') if b['page'] in pages]
    layout=dict(manifest,blocks=[b for b in manifest['blocks'] if b['page'] in pages])
    shutil.copytree(source/'fixture',out/'fixture')
    for name,data in [('views',views),('layout',layout),('buildings',controls)]:
        (out/(name+'.json')).write_text(json.dumps(data))
    print(f'{len(views)} views staged in {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--buildings',nargs='+',required=True)
    p.add_argument('--angles',nargs=2,type=float,metavar=('LEFT','RIGHT'),default=(-26,26))
    args=p.parse_args()
    stage(args.source,args.out,args.buildings,args.angles)
