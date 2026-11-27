"""Count and display local return faces and material boundaries beside missed visible wall blocks."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import read, checked, digest, save
from benchmark.texture_camera_sheet import font

RANKING=framed.REVIEW/'facade-placement-01/worst/ranking.json'
EXAMPLES={'Dev Relief 00','Relief 00','Facade','Canopy 04','Shopfront','Colonnade 01'}


def touch_pairs(ids):
    count=Counter()
    for a,b in ((ids[:-1,:],ids[1:,:]),(ids[:,:-1],ids[:,1:])):
        keep=(a!=b)&(a>0)&(b>0)
        x,y=a[keep].astype('int64'),b[keep].astype('int64')
        codes,n=np.unique((np.minimum(x,y)<<32)+np.maximum(x,y),return_counts=True)
        count.update({(int(code>>32),int(code & 0xffffffff)):int(v) for code,v in zip(codes,n)})
    return count


def score(job):
    split,frame,prior,row,run,out,title=job
    image,_,_,_,_,labels,ids,truth,_,placed,_=read_view(run,split,frame,prior,row)
    faces={f['id']:f for f in labels['faces']}
    materials={tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
    evidence=defaultdict(lambda:defaultdict(list))
    missed=truth-placed
    for (i,j),n in touch_pairs(ids).items():
        if n<4:
            continue
        for a,b in ((faces[i],faces[j]),(faces[j],faces[i])):
            p,q=tuple(a['xyz_world']),tuple(b['xyz_world'])
            if a['normal_world'][1]!=0 or p not in missed:
                continue
            axis=int(np.argmax(np.abs(a['normal_world'])))
            cross=not b['normal_world'][axis]
            distance=sum(abs(x-y) for x,y in zip(p,q))
            if cross and distance<=1:
                evidence[p]['return'].append((a['id'],b['id'],n))
            if materials[p]!=materials[q]:
                kind='coplanar_material' if a['normal_world']==b['normal_world'] and p[axis]==q[axis] else 'other_material'
                evidence[p][kind].append((a['id'],b['id'],n))
    walls={tuple(f['xyz_world']) for f in faces.values() if f['normal_world'][1]==0}
    counts=dict(wall_missed=len(missed & walls),
                adjacent_return=sum('return' in e for e in evidence.values()),
                two_return_faces=sum(len({j for _,j,_ in e.get('return',())})>=2 for e in evidence.values()),
                different_material=sum(('coplanar_material' in e or 'other_material' in e) for e in evidence.values()),
                coplanar_material=sum('coplanar_material' in e for e in evidence.values()))
    result=dict(id=frame['id'],split=split,pack_sha256=row['pack_sha256'],rgb_sha256=row['rgb_sha256'],**counts)
    if title:
        pairs=[(p,e) for p,e in evidence.items() if 'return' in e]
        def order(item):
            p,e=item
            return (bool(e.get('coplanar_material') or e.get('other_material')),
                    len({j for _,j,_ in e['return']})>=2,sum(n for _,_,n in e['return']))
        examples=[]
        for p,e in sorted(pairs,key=order,reverse=True):
            if any(sum(abs(a-b) for a,b in zip(p,s))<=2 for s,_ in examples):
                continue
            examples.append((p,e))
            if len(examples)==2:
                break
        W,H=1160,335
        page=Image.new('RGB',(W,H),'#10242d')
        draw=ImageDraw.Draw(page)
        draw.text((12,8),title,fill='white',font=font(25))
        draw.text((12,48),f"{counts['wall_missed']} missed wall blocks; {counts['adjacent_return']} touch an observed return",fill='#c6d5db',font=font(15))
        thumb=ImageOps.contain(image,(410,246))
        page.paste(thumb,(10,80))
        result['illustrated']=[]
        if not examples:
            draw.text((430,134),'No missed wall block touches a qualifying return edge in this view.',
                      font=font(19),fill='#c6d5db')
        for col,(p,e) in enumerate(examples):
            wall_ids=[f['id'] for f in faces.values() if tuple(f['xyz_world'])==p and not f['normal_world'][1]]
            side_ids=list({j for _,j,_ in e['return']})
            ys,xs=np.nonzero(np.isin(ids,wall_ids))
            l,t,r,b=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
            margin=max(18,max(b-t,r-l)//2)
            box=max(0,l-margin),max(0,t-margin),min(image.width,r+margin),min(image.height,b+margin)
            boxids=ids[box[1]:box[3],box[0]:box[2]]
            raw=image.crop(box)
            mark=np.array(raw)
            for mask,color in ((np.isin(boxids,wall_ids),(255,150,36)),(np.isin(boxids,side_ids),(0,230,245))):
                pad=np.pad(mask,1)
                inside=pad[1:-1,1:-1]&pad[:-2,1:-1]&pad[2:,1:-1]&pad[1:-1,:-2]&pad[1:-1,2:]
                mark[mask & ~inside]=color
            x=426+col*367
            draw.text((x,58),materials[p].split(':')[-1],font=font(16),fill='white')
            for offs,im in [(0,raw),(176,Image.fromarray(mark))]:
                pic=ImageOps.contain(im,(170,213),Image.Resampling.NEAREST)
                page.paste(pic,(x+offs,94))
            result['illustrated'].append(dict(xyz=list(p),material=materials[p],touch=e))
        page.save(out/(frame['id']+'.jpg'),quality=92)
    return result


def run(rundir,out,workers):
    out.mkdir(parents=True,exist_ok=True)
    meta=read(rundir/'summary.json')
    rows={r['id']:r for r in read(checked(rundir/'cases.json',meta['cases_sha256']))}
    layouts=read(RANKING)['layouts']
    targets={r['representative']['id']:r['name'] for r in layouts if r['name'] in EXAMPLES}
    jobs=[(s,f,h,rows[f['id']],rundir,out,targets.get(f['id'])) for s,f,h in framed.cases(framed.EVALUATION)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(score,jobs))
    counts={k:sum(r[k] for r in results) for k in ('wall_missed','adjacent_return','two_return_faces','different_material','coplanar_material')}
    assert len(results)==104
    cover=Image.new('RGB',(1160,68+335*len(targets)),'#10242d')
    ImageDraw.Draw(cover).text((12,12),'Source photo and native details   |   Orange: missed wall face    Cyan: touching return',font=font(23),fill='white')
    for i,key in enumerate(targets):
        with Image.open(out/(key+'.jpg')) as page:
            cover.paste(page,(0,68+335*i))
    cover.save(out/'contacts.jpg',quality=93)
    save(out/'contacts.json',dict(cases_sha256=meta['cases_sha256'],ranking_sha256=digest(RANKING),counts=counts,frames=results))
    print(counts)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    run(a.run,a.out,a.workers)
