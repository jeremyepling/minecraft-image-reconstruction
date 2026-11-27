"""Collect the skyline trials and draw source-pixel witnesses on two Colonnade views."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark import framed_depth_eval as framed,wall_section_eval
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import read,save,checked,digest
from benchmark.texture_camera_sheet import font
from benchmark.visible_skyline_opportunity import CONTROL
from engine.visible_roof_skyline import proposals,bracket


def collect(runs,out):
    out.mkdir(parents=True,exist_ok=True)
    prior=read(CONTROL/'summary.json')
    index={r['id']:r for r in read(checked(CONTROL/'cases.json',prior['cases_sha256']))}
    methods={}
    archive=out/'comparisons.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for n in ('skyline','strict','bracket','pooled','overhang'):
            root=runs/n
            meta=read(root/'summary.json')
            rows=read(checked(root/'cases.json',meta['cases_sha256']))
            assert len(rows)==200
            for r in rows:
                key=r['id']
                baseline=index[key]
                assert r['rgb_sha256']==baseline['rgb_sha256'] and r['reference_hashes']==baseline['reference_hashes']
                assert r['input_hashes']==baseline['input_hashes'] and r['control_pack_sha256']==baseline['pack_sha256']
                if r['changed']:
                    here=root/r['split']/key
                    pack=checked(here/'reconstruction.zip',r['pack_sha256'])
                    assert set(decoded_pack(pack))==set(map(tuple,read(here/'cells.json')['local']))
                    for f in ('reconstruction.zip','cells.json'):
                        z.write(here/f,f'{n}/{r["split"]}/{key}/{f}')
            transfer=runs/('transfer-'+n)
            other=read(transfer/'summary.json')
            tr=read(checked(transfer/'cases.json',other['cases_sha256']))
            assert len(tr)==4
            for r in tr:
                p=checked(transfer/r['id']/'reconstruction.zip',r['pack_sha256'])
                z.write(p,f'{n}/transfer/{r["id"]}/reconstruction.zip')
            for f in ('summary.json','cases.json'):
                z.write(root/f,f'{n}/{f}')
                z.write(transfer/f,f'{n}/transfer/{f}')
            methods[n]=dict(revision=meta['revision'],sha256=meta['adapter_sha256'],
                            groups={k:meta['groups'][k] for k in ('train','fixed','extra','new')},
                            transfer=other['relative_geometry'],canopies=meta['canopies'])
        oracle=read(runs/'opportunity.json')
        assert oracle['control_cases_sha256']==prior['cases_sha256']
        z.write(runs/'opportunity.json','oracle/cases.json')
    groups={}
    for g,splits in [('fixed',framed.EVALUATION),('development',['train']),('alternate',['extra']),('new',['new'])]:
        rows=[r for r in oracle['cases'] if r['split'] in splits]
        groups[g]={k:sum(r[k] for r in rows) for k in ('proposed','correct','clean_strips','per_cell','sections')}
    with ZipFile(archive) as z:
        assert z.testzip() is None
    save(out/'results.json',dict(control='bec7c94',cases_sha256=prior['cases_sha256'],
                                 archive_sha256=digest(archive),methods=methods,oracle=groups))
    return index


def pictures(index,out):
    pages=[]
    for split,f,h in wall_section_eval.cases(['fresh','extra']):
        if 'colonnade-01-oblique-' not in f['id']:
            continue
        image,parts,counts,camera=framed.photo(split,f,h)
        k=f['id']
        prior=set(decoded_pack(checked(CONTROL/split/k/'reconstruction.zip',index[k]['pack_sha256'])))
        ctx,pool,_=proposals(image,parts,camera,prior)
        actual,_=bracket(image,parts,counts,camera,prior)
        kept=actual-prior
        rgb=np.array(image)
        xx=np.flatnonzero(ctx.valid & ctx.union[np.minimum(ctx.skyline+2,image.height-1),np.arange(image.width)])
        rgb[ctx.skyline[xx],xx]=(239,207,76)
        xs,ys=[],[]
        for c in sorted(pool,key=lambda p:p in kept):
            u,v=ctx.face(c)
            rgb[v,u]=(20,244,225) if c in kept else (250,129,52)
            xs.extend(u);ys.extend(v)
        bounds=max(0,min(xs)-8),max(0,min(ys)-16),min(image.width,max(xs)+9),min(image.height,max(ys)+17)
        u,v=ctx.face(sorted(kept)[0])
        cx,cy=int(np.median(u)),int(np.median(v))
        zoom=(cx-34,cy-10,cx+35,cy+11)
        marked=Image.fromarray(rgb)
        pages.append((k.rsplit('-',1)[-1],image,marked.crop(bounds),len(pool),len(kept),
                      image.crop(zoom),marked.crop(zoom)))
    cover=Image.new('RGB',(1120,95+235*len(pages)),'#13242d')
    d=ImageDraw.Draw(cover)
    d.text((14,8),'Colonnade 01 | Visible roof edge above the pillars',font=font(25),fill='white')
    d.text((14,53),'Full photo at left. Enlarged roof at right. Gold: source skyline. Orange: candidate top pixels. Cyan: retained tiles.',
           font=font(16),fill='#d8e5ea')
    for i,(angle,photo,overlay,total,n,raw,marked) in enumerate(pages):
        y=95+i*235
        d.text((14,y),f'{angle.capitalize()} view',font=font(18),fill='white')
        cover.paste(ImageOps.contain(photo,(335,185)),(14,y+30))
        d.text((370,y),f'{total} candidate tiles, {n} added by the bracket rule',font=font(18),fill='white')
        pic=ImageOps.contain(overlay,(730,180),Image.Resampling.NEAREST)
        cover.paste(pic,(370,y+35))
        for left,tile,label in ((370,raw,'Native photo detail'),(735,marked,'Retained top pixels')):
            d.text((left,y+89),label,font=font(14),fill='white')
            cover.paste(tile.resize((345,105),Image.Resampling.NEAREST),(left,y+108))
    cover.save(out/'colonnade.jpg',quality=94)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('runs'))
    p.add_argument('--out',type=Path,default=Path('benchmark/reviews/visible-roof-skyline-01'))
    a=p.parse_args()
    pictures(collect(a.runs,a.out),a.out)
