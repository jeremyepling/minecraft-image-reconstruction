"""Inspect source-field and assembly evidence for the native-plane intervention."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmark.floating_stage_report import read
from benchmark.floating_wall_depth_partition import reference_parts, samples
from benchmark.expanded_baseline import digest, save
from benchmark.relative_face_assembly import axis_vector
from benchmark.relative_face_scoring import _yaw


def votes(chart, ids, levels, field):
    codes = samples(chart, ids)
    values = samples(chart, field)
    cells = {(r['u'], r['v']):(int(k), float(v)) for r,k,v in zip(chart['cells'], codes, values)}
    evidence = []
    for (x,y), (p, d) in cells.items():
        if not p:
            continue
        for u,v in ((1,0),(0,1)):
            prior = cells.get((x-u,y-v))
            if not prior or prior[0]!=p:
                continue
            for gap in (1,2,3):
                other = cells.get((x+gap*u,y+gap*v))
                if other is None or not other[0]:
                    continue
                after = cells.get((x+(gap+1)*u,y+(gap+1)*v))
                if after and after[0] == other[0]:
                    # Extrapolate each flank's one-cell local depth slope to the boundary midpoint.
                    dd = ((other[1]+.5*gap*(other[1]-after[1]))-(d+.5*gap*(d-prior[1])))
                    scaled = dd/max(.01, abs(other[1]+d)/2)
                    evidence.append((levels[other[0]]-levels[p], scaled))
                break
    return evidence


def main():
    parser=ArgumentParser(description=__doc__)
    for field in ('base','fields','output'):
        parser.add_argument('--'+field, type=Path, required=True)
    opt=parser.parse_args()
    cases=read(opt.base/'fixtures.json')['cases']
    all_rows=[]
    for case in cases:
        name=case['id']
        folder=opt.output/name
        rec=read(folder/'reference.json')
        if not rec['partition_cells']:
            continue
        trace=read(folder/'reference-offset-trace.json')['arms']['reference-offset-trace']
        score=rec['arms']['reference-offset']['score']
        assert trace['score']['geometry']==score['geometry'] and trace['score']['scores']==score['scores']
        trace_folder=folder/'reference-offset-trace'
        oriented=read(trace_folder/'oriented-charts.json')['charts']
        asm=read(trace_folder/'assembly.json')
        side=rec['details']['side']
        normal={'south':(0,0,1),'north':(0,0,-1),'east':(1,0,0),'west':(-1,0,0)}[side]
        charts=[c for c in oriented if c['id'].startswith(rec['target']+'~p') and c['id'] in asm['selected']]
        matches=[tuple(_yaw(axis_vector(c['normal_axis']),score['yaw_quarters'])) == normal for c in charts] if score['yaw_quarters'] is not None else []
        frame=Path(case['frame'])
        old=opt.base/'original'/name
        source=next(c for c in read(old/'charts.json')['charts'] if c['id']==rec['target'])
        ids,levels,_=reference_parts(source,frame)
        path=opt.fields/name/'fields/depth.npz'
        assert digest(path)==read(old/'case.json')['field_sha256']['depth']
        with np.load(path) as archive:
            field=cv2.resize(archive['value'].astype('f'),ids.shape[1::-1])
        evidence=votes(source,ids,levels,field)
        def metrics(cross):
            vals=[j*np.sign(d) if cross else abs(j) for d,j in evidence if bool(d)==cross]
            return dict(samples=len(vals),median=float(np.median(vals)) if vals else None,
                        positive=int(np.sum(np.array(vals)>0)),strong=int(np.sum(np.array(vals)>.012)))
        r=dict(id=name,split=case['split'],registered_native_normal={'checked':len(matches),'matched':sum(matches)},
               selected_siblings=len(charts),reference_connections=sum(j['kind']=='reference-depth-only' for j in asm['joins']),
               across=metrics(True),within=metrics(False))
        if name[6:9] in ('006','060'):
            rgb=np.array(Image.open(frame/'rgb.png').convert('RGB'))
            original=Image.fromarray(rgb)
            colors=np.array([[0,0,0],[0,166,220],[243,169,9],[211,67,123],[95,190,75],[145,106,230]])
            tinted=rgb.copy()
            hit=ids>0
            tinted[hit]=(.45*rgb[hit]+.55*colors[ids[hit]%len(colors)]).astype('uint8')
            gray=np.clip((field-np.quantile(field,.02))/(np.quantile(field,.98)-np.quantile(field,.02)),0,1)
            shown=Image.fromarray(cv2.applyColorMap((gray*255).astype('B'),cv2.COLORMAP_VIRIDIS)[...,::-1])
            panels=[original,Image.fromarray(tinted),shown]
            sheet=Image.new('RGB',(1200,285),'white')
            draw=ImageDraw.Draw(sheet)
            for i,(title,panel) in enumerate(zip(('Source RGB','Native same-facing depths at the chart','Source model relative inverse depth'),panels)):
                sheet.paste(panel.resize((400,250),Image.Resampling.LANCZOS),(i*400,28))
                draw.text((i*400+8,7),title,fill='black')
            sheet.save(folder/'native-and-source.png')
        all_rows.append(r)
    save(opt.output/'boundary-audit.json',all_rows)
    print('selected chart agreement',Counter((r['registered_native_normal']['checked']>0,r['registered_native_normal']['checked']==r['registered_native_normal']['matched']) for r in all_rows))
    for r in all_rows:
        if r['id'][6:9] in ('006','060'):print(r)
    print('all',[(s, sum(r[s]['samples'] for r in all_rows),sum(r[s]['positive'] for r in all_rows),sum(r[s]['strong'] for r in all_rows)) for s in ('across','within')])


if __name__=='__main__':
    main()
