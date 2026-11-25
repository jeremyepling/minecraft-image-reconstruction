"""Build a held-out, nonperiodic facade and stage its two oblique camera captures."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from benchmark.recess_control import create
from benchmark.rim_viewpoints import stage
from calibration.world import GROUND


def generate(jar,out,variant='holdout'):
    bottom=GROUND+1
    width,height=27,11
    front=1
    # Local (left, sill, width, height, depth, back). Stacked apertures and
    # varied horizontal gaps remove the original row's repeating bay pitch.
    openings=[
        (1,0,3,5,2,'tuff'), (2,7,2,2,1,'smooth_stone'),
        (6,3,4,3,1,'tuff'), (11,0,2,4,3,'andesite'),
        (12,7,5,2,2,'tuff'), (17,2,2,4,0,'polished_deepslate'),
        (21,0,4,6,2,'cobblestone'), (22,8,3,2,3,'tuff')]
    angles=(-31,29)
    if variant=='confirmation':
        width,height=29,12
        angles=(-24,33)
        openings=[
            (2,2,4,3,1,'cobbled_deepslate'), (1,8,3,2,3,'tuff'),
            (8,0,3,6,2,'tuff'), (8,8,5,3,0,'polished_deepslate'),
            (14,3,2,4,2,'smooth_stone'), (18,0,5,5,3,'tuff'),
            (20,8,2,2,1,'andesite'), (25,4,3,4,2,'diorite')]
    elif variant!='holdout':
        raise ValueError(f'Unknown entrance fixture variant: {variant}')
    regions=[];used=set()
    for index,(x,y,w,h,depth,_) in enumerate(openings):
        points={(i,j) for i in range(x,x+w) for j in range(y,y+h)}
        assert used.isdisjoint(points) and 0<x and x+w<width and 0<=y and y+h<height
        used.update(points)
        regions.append(dict(id=f'aperture-{index+1}',kind='recess',
                            bounds=[[x,bottom+y,front-depth-1],[x+w,bottom+y+h,front]]))
    name='irregular-source'
    target=[width/2,bottom+height/2,-1.5]
    source_view=dict(id=name+'-left',angle='left',kind='recess',split='test',building_id=name,
                     page=0,x=target[0]-14,feetY=target[1]+4.5-1.62,z=target[2]+27,target=target)
    building=dict(id=name,bounds=[[0,bottom,-4],[width,bottom+height,front]],regions=regions)
    # Use the existing blank-world generator and its exhaustive material/air
    # readbacks. This source record supplies the design, not an earlier facade.
    layout=dict(version='26.3',ground_y=GROUND,time=6000,weather='clear',front='south',
                client_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(),
                blocks=[dict(building_id=name,block_id='tuff',role='facade')])
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        src=Path(tmp);staging=src/'generated'
        for file,data in [('buildings',[building]),('layout',layout),('views',[source_view])]:
            (src/f'{file}.json').write_text(json.dumps(data))
        create(src,jar,staging,name,[o[4] for o in openings],[o[5] for o in openings])
        stage(staging,out,['heldout-facade'],angles=angles)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jar',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--variant',choices=('holdout','confirmation'),default='holdout')
    args=p.parse_args()
    generate(args.jar,args.out,args.variant)
