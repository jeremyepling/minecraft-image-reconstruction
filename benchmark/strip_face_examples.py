"""Count adjoining recorded faces and illustrate signed first hits in real source photos."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.camera_metrics import registration
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from benchmark.visible_error_budget import projection, stage_evidence
from benchmark.wall_return_audit import touch_pairs, RANKING


NAMES={1:'X wall',2:'top',3:'Z wall',-1:'X wall',-2:'underside',-3:'Z wall',0:'background'}
PALETTE={1:(246,165,63),-1:(246,165,63),2:(53,202,223),-2:(186,135,241),
         3:(116,204,134),-3:(116,204,134),0:(72,79,91)}
EXAMPLES={'Facade':('same_depth_error',),'Colonnade 01':('wrong_top',),'Shopfront':('exact',),
          'Canopy 04':('corner',),'Galleries':('wrong_under',),'Canopy 00 (open)':('wrong_under',)}
TITLES={'exact':'Correct wall and block','same_depth_error':'Correct wall direction, wrong depth',
        'wrong_top':'A top face rendered as a wall','wrong_under':'An underside rendered as a wall',
        'corner':'Three visible faces of one block'}


def normal_code(normal):
    axis=int(np.argmax(np.abs(normal)))
    return (axis+1)*int(np.sign(normal[axis]))


def example(kind,image,ids,faces,source,predicted,owner,xyz,world,blocks,target_name,folder):
    both=(ids!=0)&(owner>=0)
    same=np.zeros(ids.shape,bool)
    same[both]=(xyz[ids[both]]==world[owner[both]]).all(1)
    if kind=='same_depth_error':
        wall=both&(source==predicted)&(abs(source)!=2)&~same
        y,x=np.nonzero(wall)
        axis=abs(source[y,x])-1
        wall[y,x]=xyz[ids[y,x],axis]!=world[owner[y,x],axis]
        valid=wall
    elif kind in ('wrong_top','wrong_under'):
        valid=both&(source==(2 if kind=='wrong_top' else -2))&(abs(predicted)!=2)
        if kind=='wrong_top':
            roof_height=max(f['xyz_world'][1] for f in faces.values())
            upper=valid&(xyz[ids,1]==roof_height)
            if upper.any():
                valid=upper
    elif kind=='corner':
        corners=[v for v in blocks.values() if len(v)==3]
        if not corners:
            return
        group=max(corners,key=lambda group:min(faces[i]['visible_pixels'] for i in group))
        valid=np.isin(ids,group)&same
    else:
        valid=both&same&(source==predicted)&(abs(source)!=2)
    if not valid.any():
        return
    freq=np.bincount(ids[valid],minlength=len(faces)+1)
    face=int(freq.argmax())
    y,x=np.nonzero(valid&(ids==face))
    i=int(np.argmin((x-np.median(x))**2+(y-np.median(y))**2))
    px,py=int(x[i]),int(y[i])
    group=blocks[faces[face]['block']]
    ty,tx=np.nonzero(np.isin(ids,group))
    extent=max(20,min(100,max(np.ptp(tx),np.ptp(ty))*.72))
    l,t=max(0,int(px-extent)),max(0,int(py-extent))
    r,b=min(image.width,int(px+extent+1)),min(image.height,int(py+extent+1))
    box=(l,t,r,b)
    native=np.array(image.crop(box))
    a,c=source[t:b,l:r],predicted[t:b,l:r]
    def colored(normals,same_cube):
        data=native.copy()
        for code,color in PALETTE.items():
            use=normals==code
            data[use]=np.rint(.25*data[use]+.75*np.array(color))
        if kind=='corner':
            data[~same_cube]=np.rint(.35*data[~same_cube])
            padded=np.pad(same_cube,1)
            inside=padded[:-2,1:-1]&padded[2:,1:-1]&padded[1:-1,:-2]&padded[1:-1,2:]
            data[same_cube&~inside]=(255,248,212)
        return Image.fromarray(data)
    def zoom(pic):
        pic=pic.resize((228,228),Image.Resampling.NEAREST)
        p=((px-l+.5)*228/(r-l),(py-t+.5)*228/(b-t))
        pen=ImageDraw.Draw(pic)
        for radius,color in [(6,'#0b1522'),(5,'white')]:
            pen.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),outline=color,width=1)
        return pic
    whole=image.copy()
    ImageDraw.Draw(whole).rectangle(box,outline='#ffd05b',width=5)
    w,h=1190,304
    panel=Image.new('RGB',(w,h),'#122732')
    pen=ImageDraw.Draw(panel)
    pen.text((12,5),TITLES[kind]+'   |   '+target_name,font=font(22),fill='white')
    thumb=ImageOps.contain(whole,(360,225))
    panel.paste(thumb,(12,44+(225-thumb.height)//2))
    selected=owner[t:b,l:r]
    same_reference=np.isin(ids[t:b,l:r],group)
    same_rendered=(selected>=0)&(world[selected]==xyz[face]).all(2)
    for j,pic in enumerate((Image.fromarray(native),colored(a,same_reference),colored(c,same_rendered))):
        panel.paste(zoom(pic),(398+263*j,42))
    got=list(map(int,world[owner[py,px]]))
    wanted=list(map(int,xyz[face]))
    short=lambda v:'('+', '.join(map(str,v))+')'
    pen.text((397,275),'Source detail',font=font(16),fill='white')
    pen.text((661,275),NAMES[int(source[py,px])]+'  '+short(wanted),font=font(16),fill='white')
    pen.text((924,275),NAMES[int(predicted[py,px])]+'  '+short(got),font=font(16),fill='white')
    tag=target_name.lower().replace(' ','-').replace('(','').replace(')','')+'-'+kind
    panel.save(folder/(tag+'.jpg'),quality=94)
    return dict(kind=kind,photo=target_name,image=tag+'.jpg',pixel=[px,py],box=list(box),
                reference_face=face,reference_cell=wanted,rendered_cell=got,
                reference_direction=int(source[py,px]),rendered_direction=int(predicted[py,px]),
                supporting_pixels=int(freq[face]))


def measure(job):
    split,frame,inputs,row,run,folder,name=job
    image,parts,counts,camera,pose,labels,ids,truth,_,placed,_=read_view(run,split,frame,inputs,row)
    faces={f['id']:f for f in labels['faces']}
    blocks=defaultdict(list)
    codes=np.zeros(len(faces)+1,np.int8)
    xyz=np.zeros((len(faces)+1,3),int)
    for i,f in faces.items():
        blocks[f['block']].append(i)
        codes[i]=normal_code(f['normal_world'])
        xyz[i]=f['xyz_world']
    missing=truth-placed
    pairs=Counter()
    junctions=defaultdict(set)
    for (i,j),n in touch_pairs(ids).items():
        if n<4 or abs(codes[i])==abs(codes[j]):
            continue
        a,b=faces[i]['block'],faces[j]['block']
        if a==b:
            key='same_block'
            junctions[a].update((i,j))
        else:
            key='adjacent_blocks' if abs(xyz[i]-xyz[j]).sum()==1 else 'separated_blocks'
        pairs[key]+=1
    local=sorted(map(tuple,read(run/split/frame['id']/'cells.json')['local']))
    owner,axis,axes,world=projection(camera,pose,local,row['translation'],ids.shape)
    rank=stage_evidence(parts,counts,image.size,camera,axes,abs(codes[ids])-1)
    right_pixels=np.bincount(ids[rank>=2],minlength=len(codes))
    metrics=defaultdict(Counter)
    for block,group in blocks.items():
        coord=tuple(xyz[group[0]])
        q=metrics[len(group)]
        q['visible']+=1
        miss=coord in missing
        q['missed']+=int(miss)
        joint=len(junctions[block])>=2
        q['shared_edge']+=int(joint)
        q['missed_shared_edge']+=int(miss and joint)
        supported=sum(right_pixels[i]*2>=faces[i]['visible_pixels'] for i in group)
        q['missed_with_two_detected_axes']+=int(miss and supported>=2)
    examples=[]
    if name in EXAMPLES:
        p=np.asarray(camera['P'])
        t=registration(p,pose)
        eye=-np.linalg.solve(p[:,:3],p[:,3])
        hit=owner>=0
        a=axis[hit]
        sign=np.where(eye[a]>np.asarray(local)[owner[hit],a],1,-1)
        rendered=np.zeros(ids.shape,np.int8)
        rendered[hit]=(axes[a]+1)*sign*np.sign(t[axes[a],a]).astype(int)
        for kind in EXAMPLES[name]:
            chosen=example(kind,image,ids,faces,codes[ids],rendered,owner,xyz,world,blocks,name,folder)
            if chosen:
                examples.append(chosen)
    return dict(split=split,id=frame['id'],rgb=row['rgb_sha256'],pack=row['pack_sha256'],
                blocks=metrics,perpendicular_contacts=pairs,examples=examples)


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    meta=read(args.run/'summary.json')
    index={r['id']:r for r in read(checked(args.run/'cases.json',meta['cases_sha256']))}
    names={r['representative']['id']:r['name'] for r in read(RANKING)['layouts']}
    jobs=[(s,f,h,index[f['id']],args.run,args.out,names.get(f['id'])) for s,f,h in framed.cases(framed.EVALUATION)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(measure,jobs))
    totals=defaultdict(Counter)
    contacts=Counter()
    choices={}
    for r in rows:
        contacts.update(r['perpendicular_contacts'])
        for k,c in r['blocks'].items():
            totals[k].update(c)
        for ex in r['examples']:
            k=ex['kind']
            if k not in choices or choices[k]['supporting_pixels']<ex['supporting_pixels']:
                choices[k]=ex
    assert sum(c['visible'] for c in totals.values())==23702 and sum(c['missed'] for c in totals.values())==3167
    sheet=Image.new('RGB',(1190,117+304*len(choices)),'#122732')
    draw=ImageDraw.Draw(sheet)
    draw.text((12,6),'Which face does the camera see first?',font=font(29),fill='white')
    draw.text((12,48),'Amber: X wall      Green: Z wall      Cyan: top      Violet: underside      Gray: background',font=font(19),fill='#dfe7e9')
    draw.text((12,83),'Full screenshot',font=font(17),fill='white')
    for j,text in enumerate(('Photographed pixels','Recorded face and block','Reconstructed face and block')):
        draw.text((398+263*j,83),text,font=font(16),fill='white')
    for j,k in enumerate(k for k in TITLES if k in choices):
        with Image.open(args.out/choices[k]['image']) as panel:
            sheet.paste(panel,(0,117+304*j))
    sheet.save(args.out/'face-directions.jpg',quality=94)
    compact=Image.new('RGB',(1190,117+2*304),'#122732')
    compact.paste(sheet.crop((0,0,1190,117)),(0,0))
    for j,k in enumerate(('same_depth_error','wrong_top')):
        with Image.open(args.out/choices[k]['image']) as panel:
            compact.paste(panel,(0,117+304*j))
    compact.save(args.out/'depth-and-direction.jpg',quality=95)
    save(args.out/'face-contacts.json',dict(source_cases_sha256=digest(args.run/'cases.json'),frames=rows,
                                           totals=totals,perpendicular_contacts=contacts,examples=choices))
    print(dict(totals),contacts,list(choices))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('../visible-error-budget/runs/primary-production'))
    p.add_argument('--out',type=Path,default=Path('benchmark/reviews/strip-orientation-01/faces'))
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
