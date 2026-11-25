"""Generate building-disjoint shape and adjacency calibration scenes in a fresh flat world."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from zipfile import ZipFile, ZIP_DEFLATED

from calibration.world import GROUND, PACK, create_world

MATERIALS = ('tuff','cobblestone','andesite','deepslate', *('dead_'+s+'_coral_block' for s in ('brain','bubble','fire','horn','tube')))
SPLITS = ('train','dev','test')
WALL = ('stone_bricks','andesite','tuff')
UP = GROUND+1


def clear_sightline(eye, endpoint, occupied, ground_y=GROUND):
    """Test a ground-truth ray up to, but excluding, a feature's surface endpoint."""
    delta=[b-a for a,b in zip(eye,endpoint)]
    cell=[math.floor(v) for v in eye]
    step=[1 if d>0 else -1 if d<0 else 0 for d in delta]
    increment=[1/abs(d) if d else math.inf for d in delta]
    crossing=[((c+(s>0)-p)/d if d else math.inf) for c,s,p,d in zip(cell,step,eye,delta)]
    while True:
        if cell[1]<=ground_y or tuple(cell) in occupied:
            return False
        near=min(crossing)
        if near>=1-1e-7:
            return True
        for i in range(3):
            if crossing[i]<=near+1e-10:
                cell[i]+=step[i]
                crossing[i]+=increment[i]


class Building:
    def __init__(self,page,split,kind,parameters):
        self.origin=page*96
        self.cells={}
        self.data=dict(id=f'{split}-{kind}-{page:02}',page=page,split=split,kind=kind,parameters=parameters,
                       measurements=[],probes=[],regions=[])
        self.material=WALL[SPLITS.index(split)]

    def xyz(self,p):
        return [p[0]+self.origin,p[1]+UP,p[2]]

    def cuboid(self,lo,hi,role,block=None):
        for x in range(lo[0],hi[0]):
            for y in range(lo[1],hi[1]):
                for z in range(lo[2],hi[2]):
                    self.cells[x,y,z]=(block or self.material,role)

    def measure(self,name,a,b):
        axes=[i for i in range(3) if a[i]!=b[i]]
        assert len(axes)==1
        self.data['measurements'].append(dict(id=name,a=self.xyz(a),b=self.xyz(b),value=abs(b[axes[0]]-a[axes[0]]),axis='xyz'[axes[0]]))

    def probe(self,name,p,occupied,role):
        assert (tuple(math.floor(v) for v in p) in self.cells)==occupied,(name,p)
        self.data['probes'].append(dict(id=name,point=self.xyz(p),occupied=occupied,role=role))

    def region(self,name,kind,lo,hi):
        self.data['regions'].append(dict(id=name,kind=kind,bounds=[self.xyz(lo),self.xyz(hi)]))

    def finish(self):
        # Separate exposed faces by depth, direction and structural role. Cover each
        # plane with disjoint rectangles. A recess opening never becomes a face.
        groups=defaultdict(set)
        for pos,(_,role) in self.cells.items():
            for axis in range(3):
                u,v=(axis+1)%3,(axis+2)%3
                for sign in [-1,1]:
                    n=list(pos); n[axis]+=sign
                    if tuple(n) in self.cells or (axis==1 and n[1]<0):
                        continue
                    groups[axis,sign,pos[axis]+(sign==1),role].add((pos[u],pos[v]))
        planes=[]
        for (axis,sign,depth,role),uv in sorted(groups.items()):
            u,v=(axis+1)%3,(axis+2)%3
            while uv:
                a,b=min(uv); w=h=1
                while (a+w,b) in uv:
                    w+=1
                while all((i,b+h) in uv for i in range(a,a+w)):
                    h+=1
                uv.difference_update((i,j) for i in range(a,a+w) for j in range(b,b+h))
                vertices=[]
                for s,t in [(a,b),(a+w,b),(a+w,b+h),(a,b+h)]:
                    p=[0,0,0]; p[axis]=depth; p[u]=s; p[v]=t
                    vertices.append(self.xyz(p))
                normal=[0,0,0];normal[axis]=sign
                planes.append(dict(id=f'plane-{len(planes)}',role=role,normal=normal,
                                   vertices=vertices if sign==1 else vertices[::-1]))
        lo=[min(p[i] for p in self.cells) for i in range(3)]
        hi=[max(p[i] for p in self.cells)+1 for i in range(3)]
        self.data.update(planes=planes,bounds=[self.xyz(lo),self.xyz(hi)])
        return lo,hi


def canopy(b,gap,over,k,open_back=False):
    height=5+k
    width=gap+2
    for x in [0,width-1]:
        b.cuboid((x,0,0),(x+1,height,1),'pillar')
    if not open_back:
        b.cuboid((0,0,-4),(width,height,-3),'back_wall')
    else:
        for x in [0,width-1]:
            b.cuboid((x,0,-4),(x+1,height,-3),'rear_pillar')
    b.cuboid((-1,height,-5),(width+1,height+1,1+over),'roof')
    b.measure('clear_pillar_spacing',(1,1,1),(width-1,1,1))
    b.measure('roof_overhang',(0,height,1),(0,height,1+over))
    b.measure('clear_height',(0,0,1),(0,height,1))
    b.measure('roof_width',(-1,height,1+over),(width+1,height,1+over))
    b.probe('under_canopy',(width/2,2.5,.5),False,'air')
    b.probe('front_post',(.5,2.5,.5),True,'pillar')
    b.probe('roof_edge',(.5,height+.5,over+.5),True,'roof')


def tower(b,k):
    # Two independent ledges on both side and front; all buildings total 15 high.
    levels=[(0,5,0,0),(5,10,1+k,1+k%2),(10,15,3+k,3+k)]
    for i,(low,high,side,back) in enumerate(levels):
        b.cuboid((side,low,-8),(16-side,high,1-back),f'tier_{i}')
        b.measure(f'tier_{i}_width',(side,low,1-back),(16-side,low,1-back))
        if i:
            prev=levels[i-1]
            b.measure(f'front_setback_{i}',(side,low,1-prev[3]),(side,low,1-back))
            b.measure(f'side_setback_{i}',(prev[2],low,1-prev[3]),(side,low,1-prev[3]))
            b.probe(f'ledge_air_{i}',(prev[2]+.5,low+.5,.5-prev[3]),False,'air')
    b.measure('total_height',(8,0,-8),(8,15,-8))


def recess(b,k):
    depths=[1,2,3] if k!=1 else [2,3,1]
    width=25
    b.cuboid((0,0,-4),(width,8,1),'facade')
    for i in range(6):
        left=1+4*i; low=0 if i%2==0 else 3
        high=5 if not low else 6
        depth=depths[(i//2+k)%3]
        for x in range(left,left+3):
            for y in range(low,high):
                for z in range(1-depth,1):
                    del b.cells[x,y,z]
                b.cells[x,y,-depth]=(b.material,'recess_back')
        name=('door' if low==0 else 'window')+str(i//2+1)
        b.measure(name+'_depth',(left,low+1,1),(left,low+1,1-depth))
        b.region(name,'recess',(left,low,-depth),(left+3,high,1))
        b.probe(name+'_air',(left+1.5,low+.5,.5),False,'air')
        b.probe(name+'_back',(left+1.5,low+.5,.5-depth),True,'recess_back')


def corners(b,k):
    # Same-size regions on opposite sides of a clear gap. Only the right seam turns.
    w=6+k; h=7
    b.cuboid((0,0,0),(w,h,1),'flat_a','andesite')
    b.cuboid((w,0,0),(2*w,h,1),'flat_b','tuff')
    b.region('flat_seam','coplanar_seam',(0,0,0),(2*w,h,1))
    x=2*w+4
    b.cuboid((x,0,0),(x+w,h,1),'corner_front','andesite')
    b.cuboid((x+w-1,0,1-w),(x+w,h,0),'corner_return','andesite')
    b.region('right_angle','geometric_corner',(x,0,1-w),(x+w,h,1))
    b.measure('flat_seam_line',(w,0,1),(w,h,1))
    b.measure('geometric_corner_line',(x+w,0,1),(x+w,h,1))
    b.probe('corner_air',(x+w-1.5,2.5,-.5),False,'air')
    b.probe('corner_return',(x+w-.5,2.5,-.5),True,'corner_return')


def adjacency(b,k):
    rng=random.Random([413,125,331][k])
    aa=list(MATERIALS);rng.shuffle(aa)
    for y,a in enumerate(aa):
        others=list(MATERIALS);rng.shuffle(others)
        for col,c in enumerate(others):
            x=col*4
            for dx,mat in enumerate((a,a,c)):
                b.cells[x+dx,y,0]=(mat,'adjacency_wall')
            b.probe(f'air_{col}_{y}',(x+3.5,y+.5,.5),False,'air')
    b.cuboid((0,9,-1),(17,10,5),'shade_roof','smooth_stone')
    b.region('under_roof','shade',(0,0,0),(17,9,1))
    b.region('open_sky','sun',(18,0,0),(35,9,1))


def generate(jar_path,out,open_only=False):
    out.mkdir(parents=True,exist_ok=True)
    world=out/'fixture/saves/Palette Lab'
    if world.exists():
        raise FileExistsError(world)
    with ZipFile(jar_path) as jar:
        meta=json.loads(jar.read('version.json'))
    if meta['id']!='26.3':
        raise ValueError('Expected a 26.3 client')
    create_world(world,meta,{'minecraft:random_tick_speed':0})
    (world/'datapacks').mkdir()
    blocks=[];records=[];views=[]
    with ZipFile(world/'datapacks'/PACK,'w',ZIP_DEFLATED) as pack:
        fmt=[meta['pack_version'][s] for s in ('data_major','data_minor')]
        pack.writestr('pack.mcmeta',json.dumps({'pack':dict(description='Geometry calibration buildings',min_format=fmt,max_format=fmt)}))
        for k,split in enumerate(SPLITS):
            # Adjacency first so the texture arm can begin extraction immediately.
            cases=[('adjacency',{}),('stepped_tower',{}),('recess',{}),('corner',{})]
            cases += [('canopy',dict(gap=g,overhang=o)) for g in (2,4,6) for o in (1,2,4)]
            if open_only:
                cases=[('canopy',dict(gap=g,overhang=o,open_back=True)) for g,o in zip((2,4,6),(1,2,4))]
            for kind,params in cases:
                page=len(records);b=Building(page,split,kind,params)
                if kind=='canopy':
                    canopy(b,params['gap'],params['overhang'],k,open_only)
                else:
                    {'adjacency':adjacency,'stepped_tower':tower,'recess':recess,'corner':corners}[kind](b,k)
                lo,hi=b.finish()
                from benchmark.geometry_metrics import enriched
                records.append(enriched(b.data))
                commands=[]
                for p,(mat,role) in sorted(b.cells.items()):
                    x,y,z=b.xyz(p)
                    blocks.append(dict(x=x,y=y,z=z,block_id=mat,state=mat,side_texture=mat,page=page,split=split,
                                       building_id=b.data['id'],kind=kind,role=role,invisible=False))
                    commands.append(f'setblock {x} {y} {z} minecraft:{mat} strict')
                pack.writestr(f'data/palette_lab/function/page_{page}.mcfunction','\n'.join(commands)+'\n')
                cx,cy,cz=b.xyz([(a+c)/2 for a,c in zip(lo,hi)])
                dist=max(12,(hi[1]-lo[1])*1.55,(hi[0]-lo[0])*.95)+(hi[2]-lo[2])*.45
                for angle,az,el in [('front',0,7),('left',-32,12),('right',35,12),('elevated',-18,34)]:
                    az,el=map(math.radians,[az,el])
                    views.append(dict(id=f"{b.data['id']}-{angle}",page=page,split=split,kind=kind,building_id=b.data['id'],
                                      angle=angle,x=cx+dist*math.sin(az)*math.cos(el),feetY=cy+dist*math.sin(el)-1.62,
                                      z=cz+dist*math.cos(az)*math.cos(el),target=[cx,cy,cz]))
    layout=dict(version='26.3',ground_y=GROUND,time=6000,weather='clear',front='south',blocks=blocks,
                client_sha256=hashlib.sha256(jar_path.read_bytes()).hexdigest())
    for name,data in [('layout',layout),('views',views),('buildings',records)]:
        (out/f'{name}.json').write_text(json.dumps(data,separators=(',',':')))
    assert len(views)==(36 if open_only else 156) and len(records)==(9 if open_only else 39)
    print(len(records),'buildings;',len(blocks),'solid voxels;',len(views),'views')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jar',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--open-only',action='store_true')
    args=p.parse_args()
    generate(args.jar,args.out,args.open_only)
