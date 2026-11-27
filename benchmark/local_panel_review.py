"""Collect decoded paired scores, panel-oracle evidence and an annotated relief case."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed, wall_section_eval as frozen
from benchmark.local_panel_trial import CONTROL
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, read, save, digest
from benchmark.texture_camera_sheet import font
from benchmark.wall_source_projection import Scene

OUT=Path('benchmark/reviews/local-panel-depth-01')
RUN=Path('runs')
KEY='signed-relief-26.3--dev--dev-relief-00-oblique-right'
TRIALS={'coherent':['dev-scores','scores','shelter-gate'],'per_block':['guard-dev','guard','guard-transfer'],
        'retain':['retained','retained-transfer'],'phase':['atlas','atlas-transfer'],
        'verified':['verified','verified-transfer']}


def highlight(image,scene,cells,color):
    ids=[scene.cells.index(c) for c in cells]
    mask=np.isin(scene.owner,ids).reshape(scene.shape)
    mask=np.array(Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST))
    pad=np.pad(mask,1)
    edge=mask & ~(pad[1:-1,:-2]&pad[1:-1,2:]&pad[2:,1:-1]&pad[:-2,1:-1])
    rgb=np.array(image).copy()
    rgb[mask]=(rgb[mask]*.83+np.asarray(color)*.17).astype(np.uint8)
    rgb[edge]=color
    return Image.fromarray(rgb),mask


def illustrate(row):
    s,f,p=next(job for job in frozen.cases(framed.EVALUATION) if job[1]['id']==KEY)
    image,parts,_,cam=framed.photo(s,f,p)
    old=set(decoded_pack(checked(CONTROL/s/KEY/'reconstruction.zip',row['control_pack_sha256'])))
    new=set(decoded_pack(checked(RUN/'scores'/s/KEY/'reconstruction.zip',row['pack_sha256'])))
    orig,mask1=highlight(image,Scene(image,parts,cam,old),old-new,(255,153,54))
    moved,mask2=highlight(image,Scene(image,parts,cam,new),new-old,(43,239,218))
    y,x=np.nonzero(mask1|mask2)
    pad=max(20,(y.max()-y.min())//2,(x.max()-x.min())//2)
    box=(max(0,x.min()-pad),max(0,y.min()-pad),min(image.width,x.max()+1+pad),min(image.height,y.max()+1+pad))
    marked=image.copy()
    ImageDraw.Draw(marked).rectangle(box,outline=(255,227,136),width=3)
    card=Image.new('RGB',(1260,388),'#12262e')
    d=ImageDraw.Draw(card)
    d.text((14,10),'Dev Relief 00   |   Correcting the depth of a two-block panel',font=font(27),fill='white')
    for j,(label,im) in enumerate([('Source photograph',marked),('Source detail',image.crop(box)),
                                    ('Previous placements',orig.crop(box)),('Shifted placements',moved.crop(box))]):
        x0=14+310*j
        d.text((x0,62),label,font=font(17),fill='white')
        im=ImageOps.contain(im,(298,235),Image.Resampling.NEAREST)
        card.paste(im,(x0,94+(235-im.height)//2))
    d.text((14,350),'Orange and teal outline visible pixels owned by the two exchanged blocks. Both teal positions match the recorded visible surface.',font=font(17),fill='#c8d7dc')
    card.save(OUT/'dev-relief.jpg',quality=94)


def collect():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=read(CONTROL/'summary.json')
    base={(r['split'],r['id']):r for r in read(checked(CONTROL/'cases.json',baseline['cases_sha256']))}
    names={r['representative']['id']:r['name'] for r in read(framed.REVIEW/'facade-placement-01/worst/ranking.json')['layouts']}
    groups={}
    named={}
    control_hash=baseline['cases_sha256']
    with ZipFile(OUT/'scores.zip','w',ZIP_DEFLATED) as z:
        for method,paths in TRIALS.items():
            summary={}
            count=0
            for name in paths:
                root=RUN/name
                meta=read(root/'summary.json')
                records=read(checked(root/'cases.json',meta['cases_sha256']))
                if 'groups' in meta:
                    summary.update({k:g for k,g in meta['groups'].items() if k in ('train','fixed','extra','new')})
                else:
                    summary['transfer']=meta['relative_geometry']
                for r in records:
                    count+=1
                    if 'split' not in r:
                        pack=checked(root/r['id']/'reconstruction.zip',r['pack_sha256'])
                    else:
                        base_row=base[r['split'],r['id']]
                        assert base_row['rgb_sha256']==r['rgb_sha256'] and base_row['input_hashes']==r['input_hashes']
                        assert base_row['pack_sha256']==r['control_pack_sha256'] and base_row['reference_hashes']==r['reference_hashes']
                        if r['id'] in names:
                            named.setdefault(names[r['id']],{})[method]=r['relative_geometry']
                        if r['id']==KEY and method=='coherent':
                            illustrate(r)
                        if r['id']==KEY and method=='verified':
                            observed=r['relative_geometry']
                            assert tuple(observed[k] for k in ('gained','lost','extra_added','extra_removed','hidden_removed'))==(2,0,0,2,2)
                        if not r['changed']:
                            continue
                        where=root/r['split']/r['id']
                        pack=checked(where/'reconstruction.zip',r['pack_sha256'])
                        assert set(decoded_pack(pack))==set(map(tuple,read(where/'cells.json')['local']))
                    z.write(pack,f'{name}/{r["id"]}/reconstruction.zip')
                for f in ('summary.json','cases.json'):
                    z.write(root/f,f'{name}/{f}')
            assert count==204
            groups[method]=summary
        oracle={}
        for run in ('development','evaluation'):
            path=RUN/run
            meta=read(path/'summary.json')
            assert meta['control_cases_sha256']==control_hash
            rows=read(checked(path/'cases.json',meta['cases_sha256']))
            for r in rows:
                k='fixed' if r['split'] in framed.EVALUATION else r['split']
                g=oracle.setdefault(k,Counter())
                g.update(frames=1,options=r['options'],correct=r['oracle'],frames_with_clean=int(r['oracle']>0))
                z.write(path/r['split']/r['id']/'options.json',f'options/{r["id"]}.json')
                if r['id'] in names:
                    named[names[r['id']]].update(oracle=r['oracle'],options=r['options'])
            for f in ('summary.json','cases.json'):
                z.write(path/f,f'options/{run}/{f}')
        camera={}
        for k in ('side','elevated','native'):
            root=RUN/('verified-'+k)
            original=Path('../narrow-post-count/runs')/(k+'-production')
            cm=read(original/'summary.json')
            bm=read(root/'summary.json')
            old={r['id']:r for r in read(checked(original/'cases.json',cm['cases_sha256']))}
            current=read(checked(root/'cases.json',bm['cases_sha256']))
            assert len(old)==len(current)
            for r in current:
                before=old[r['id']]
                assert all(r[key]==before[key] for key in ('rgb_sha256','reference_hashes','input_hashes','pack_sha256'))
            camera[k]=dict(views=len(current),changed=0,camera_supported=sum(r['camera_supported'] for r in current))
            for f in ('summary.json','cases.json'):
                z.write(root/f,'camera/'+k+'/'+f)
        for f in ('timing.json','pillar-input.json'):
            z.write(RUN/f,f)
    with ZipFile(OUT/'scores.zip') as z:
        assert z.testzip() is None
    save(OUT/'results.json',dict(control='bec7c94',geometry_control='028819a',control_cases_sha256=control_hash,
                                 archive_sha256=digest(OUT/'scores.zip'),results=groups,oracle=oracle,named=named,
                                 camera=camera,timing=read(RUN/'timing.json'),pillar=read(RUN/'pillar-input.json')))


if __name__=='__main__':
    collect()
