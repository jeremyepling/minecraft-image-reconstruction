"""Collect finite options, native support and decoded-pair scores without changing choices."""
from hashlib import sha256
from io import BytesIO
import gzip
from zipfile import ZipFile
from benchmark.return_face_exports import incumbent
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from benchmark.return_face_round import CONTROL, PC, RUN, ROOT, read, save, digest, source
from benchmark.expanded_scores import pooled
from benchmark.return_face_proposals import observed_support

REVIEW=ROOT/'benchmark/reviews/return-face-proposals-01'
CAPACITY={'fixed-104':104,'visible-builds-v3':144,'decorative-builds-v1':40}


def coverage(native,options,base):
    """Native level correctness of best independently chosen option per RGB component."""
    current=best=unbounded=total=0
    for key,rec in native.items():
        hist=rec['histogram']
        total+=sum(hist.values())
        current+=hist.get(str(base),0)
        unbounded+=max(hist.values(),default=0)
        levels={base,*[o['target_level'] for o in options if o['group']==int(key)]}
        best+=max(hist.get(str(v),0) for v in levels)
    return dict(native_same_axis=total,incumbent=current,best_offered=best,unrestricted=unbounded)


def card(row,text,which):
    image,parts,_,_=source(row)
    data=read(RUN/row['id']/'source.json')
    anchor=read(RUN/row['id']/'ideal-anchors.json')
    parent=next(p for p in parts if p['id']==data['parent'])
    shape=image.height//4,image.width//4
    def overlay(mask,color):
        nonlocal image
        mask=np.array(Image.fromarray(mask.astype('uint8')).resize(image.size,Image.Resampling.NEAREST),bool)
        rgb=np.asarray(image).copy()
        rgb[mask]=.55*rgb[mask]+.45*np.array(color)
        image=Image.fromarray(rgb)
    overlay(observed_support(parent['support'],shape,image.size),(244,177,30))
    if which=='oracle':
        for key in {o['anchor'] for o in anchor['options']}:
            part=next((p for p in parts if p['id']==key),None)
            if part: overlay(observed_support(part['support'],shape,image.size),(229,31,153))
    if which=='chosen':
        choice=data['chosen']
        mask=np.zeros(shape,bool)
        for y,a,b in read(RUN/row['id']/'masks.json')[str(choice['group'])]:
            mask[y,a:b]=True
        overlay(mask,(52,222,231))
        support=next(p for p in parts if p['id']==choice['anchor'])
        overlay(observed_support(support['support'],shape,image.size),(224,65,157))
    image.thumbnail((720,450))
    result=Image.new('RGB',(720,505),'#17232d')
    result.paste(image,(0,50))
    draw=ImageDraw.Draw(result)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    draw.text((10,4),row['id'],font=font,fill='white')
    draw.text((10,26),text,font=font,fill='#d7dee2')
    return result


def canonical_pack(payload):
    with ZipFile(BytesIO(payload)) as archive:
        result={}
        for name in archive.namelist():
            content=archive.read(name)
            if name.endswith('.nbt'):
                content=gzip.decompress(content)
            result[name]=sha256(content).hexdigest()
        return result


def main():
    receipt=read(CONTROL)
    rows=receipt['pilot']['frames']
    cases=[]
    for row in rows:
        folder=RUN/row['id']
        chosen=read(folder/'chosen/score.json')
        forced=read(folder/'forced/score.json') if (folder/'forced/score.json').exists() else None
        src=read(folder/'source.json')
        ideal=read(folder/'ideal-anchors.json')
        native=read(folder/'grade.json')
        candidates=[('incumbent',row['baseline']),('chosen',chosen['case'])]
        if forced: candidates.append(('forced',forced['case']))
        winner,case=max(candidates,key=lambda item:item[1]['geometry']['strict']['f1'])
        contents=canonical_pack((folder/'chosen/reconstruction.zip').read_bytes())
        assert contents==canonical_pack(incumbent(row))
        cases.append(dict(canonical_members=contents,id=row['id'],dataset=row['dataset'],split=row['split'],rgb_sha256=row['rgb']['sha256'],
             source_pack_sha256=chosen['incumbent_sha256'],native_ids_sha256=row['visible_ids_sha256'],
             incumbent=row['baseline'],source=src,ideal_return=ideal,native=native,
             native_coverage=coverage(native,src['options'],src['initial']),
             oracle_anchor_coverage=coverage(native,ideal['options'],src['initial']),
             chosen=chosen,forced=forced,best_export=winner,oracle_case=case))
    datasets={}
    for dataset,available in [*CAPACITY.items(),('pilot',13)]:
        group=[c for c in cases if dataset=='pilot' or c['dataset']==dataset]
        datasets[dataset]=dict(incumbent=pooled([c['incumbent'] for c in group],available),
            chosen=pooled([c['chosen']['case'] for c in group],available),
            best_export=pooled([c['oracle_case'] for c in group],available))
    splits={}
    for name,group in [('dev',cases[:4]),('harbor_castle',cases[4:9]),('fixed',cases[9:11]),('decorative',cases[11:])]:
        splits[name]=pooled([c['chosen']['case'] for c in group],len(group))
    save(REVIEW/'cases.json.gz',cases)
    save(REVIEW/'summary.json',dict(base='e3bd220',atlas_version='26.3',control_sha256=digest(CONTROL),datasets=datasets,splits=splits,identical_canonical_archives=len(cases),
            cases=[dict(id=c['id'],source_sha256=c['rgb_sha256'],incumbent_sha256=c['source_pack_sha256'],
             pack_sha256=c['chosen']['pack_sha256'],paired=c['chosen']['paired'],best_export=c['best_export'],
             forced=c['forced']['paired'] if c['forced'] else None,native=c['native_coverage'],
             oracle_anchor_native=c['oracle_anchor_coverage']) for c in cases]))
    sheet=Image.new('RGB',(1440,505))
    sheet.paste(card(rows[6],'Gold wall · pink returns identified by reference anchor diagnostic','oracle'),(0,0))
    sheet.paste(card(rows[9],'Gold wall · cyan chosen component · pink ground-anchored return','chosen'),(720,0))
    sheet.save(REVIEW/'source-returns.jpg',quality=90)
    for k,d in datasets.items():
        print(k,[(a,round(v['geometry']['strict']['f1']*100,3)) for a,v in d.items()])
    for c in cases:
        print(c['id'],c['best_export'],c['chosen']['paired'],c['forced'] and c['forced']['paired'])

if __name__=='__main__':
    main()
