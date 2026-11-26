"""Join the independent counting trials and render one scored example per building."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_sheet import label
from benchmark.crop_fit_sheet import size_text
from benchmark.detail_round import support
from benchmark.normal_round import groups
from benchmark.strip_review_sheet import overlay
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from benchmark.surface_reference import recorded
from benchmark.texture_count_round import FINDERS, summarize
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only

BASE = ROOT/'benchmark/reviews/texture-photo-count-01'
METHODS = ['current', 'self_repeat', 'phase', 'coplanar']


def indexed(records, field):
    table = {r[field]: r for r in records}
    if len(table) != len(records):
        raise ValueError('Repeated '+field)
    return table


def join(phase, coplanar):
    base = read(checked(BASE/'cases.json', read(BASE/'manifest.json')['artifacts']['cases.json']))
    for method, (folder, field) in zip(('phase','coplanar'), (phase,coplanar)):
        rows = indexed(read(folder/'cases.json'), 'key')
        if rows.keys() != indexed(base,'key').keys():
            raise ValueError('Frame set changed')
        for frame in base:
            row = rows[frame['key']]
            if row['sha256'] != frame['sha256']:
                raise ValueError('Screenshot mismatch')
            candidates = indexed(row['surfaces'],'proposal')
            if candidates.keys() != indexed(frame['surfaces'],'proposal').keys():
                raise ValueError('Surface set changed')
            for part in frame['surfaces']:
                result = candidates[part['proposal']]
                if result['guess']['current'] != part['guess']['current']:
                    raise ValueError('Baseline size mismatch')
                part['guess'][method] = result['guess'][field]
                if method=='phase':
                    part['guess']['self_repeat'] = result['guess']['self_repeat']
    for frame in base:
        frame.pop('processing_s',None)
        frame.pop('predictions',None)
        frame['scores'] = summarize([frame],METHODS)
        for part in frame['surfaces']:
            part['guess'] = {k:part['guess'][k] for k in METHODS}
    return base


def examples(rows):
    """Favor changed counts; prefer a different outcome in each successive column."""
    selected = {}
    seen = Counter()
    for layout in sorted({row['layout'] for row in rows}):
        options = []
        for row in rows:
            if row['layout'] != layout:
                continue
            for part in row['surfaces']:
                if 'actual' not in part:
                    continue
                guesses = part['guess']
                changes = [(method, guesses[method] == part['actual']) for method in METHODS[1:]
                           if guesses[method] != guesses['current']]
                portico_beam = layout=='portico' and part['actual']==[18,1] and guesses['self_repeat']==part['actual'] and guesses['phase']!=part['actual']
                priority = (portico_beam,bool(changes), sum(1/(1+seen[k]) for k in changes),
                            None not in guesses['current'],np.prod(part['actual']))
                options.append((priority, row, part, changes))
        _, row, part, changes = max(options,key=lambda entry: entry[0])
        selected[layout] = row,part
        seen.update(changes)
    return selected


def render(rows, out):
    from benchmark.phase_corners import predict_surface
    from benchmark.coplanar_count import predict_frame

    source,method = FINDERS['photo']
    manifest = read(source/'manifest.json')
    meta = indexed(read(checked(source/'cases.json',manifest['artifacts']['cases.json'])),'key')
    caps = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    selected = examples(rows)
    stages = ['Screenshot','Surface Finder','Neighboring pixels','Straightened crop','Default count',
              'Texture-guided crop','Repeats alone','With atlas check','Possible donor','Same-plane count','Recorded surface','Score']
    w,h,left,top = 278,275,190,118
    page = Image.new('RGB',(left+w*len(selected),top+h*len(stages)),'#f6f5ef')
    draw = ImageDraw.Draw(page)
    label(draw,(18,12),'Can another texture tell us the size of a surface?',34,bold=True)
    label(draw,(18,62),'One example per building. Teal shows predicted support. Gold in the recorded row is the visible reference. A ? marks an unresolved axis.',21)
    for i,stage in enumerate(stages):
        y = top+i*h
        draw.line((8,y,page.width-8,y),fill='#aab6b3')
        label(draw,(12,y+8),stage,17,bold=True)
    index = {}
    for col,(layout,(frame,rec)) in enumerate(selected.items()):
        key = frame['key']
        cap = caps[key]
        with Image.open(checked(cap.path('image'), frame['sha256'])) as im:
            photo = im.convert('RGB')
        path = meta[key]['predictions']
        parts = read_gzip(checked(source/path,manifest['artifacts'][path]))[method]
        samples = [sample_surface(photo,p) for p in parts]
        base = [count(s) for s in samples]
        idx = next(i for i,p in enumerate(parts) if p['id']==rec['proposal'])
        part,sample = parts[idx],samples[idx]
        test = predict_surface(photo,part,sample,deepcopy(base[idx]))
        transfer = predict_frame(photo,parts,samples,deepcopy(base))[idx]
        pose,ref = recorded(cap)
        native = _native_reference(pose,ref,read(cap.path('labels'))['faces'])
        truth = native==rec['reference']
        mask = support(part,photo.size)
        l,t,r,b = Image.fromarray(mask|truth).getbbox()
        box = (max(0,l-12),max(0,t-12),min(photo.width,r+12),min(photo.height,b+12))
        locator=photo.copy()
        ImageDraw.Draw(locator).rectangle(box,outline='white',width=3)
        cells = [(locator, 'White box marks surface'),
                 (overlay(photo,box,[(mask,(26,218,205),.62)]),'Image-only outline'),
                 (photo.crop(box),'Original nearby pixels')]
        repeat=test.get('evidence',{}).get('self_repeat',base[idx])
        for name,result in zip(METHODS,(base[idx],repeat,test,transfer)):
            chart = sample_surface(photo,dict(part,quad=result['quad'])) if result.get('quad') is not None else sample
            guess = dimensions(part,chart['quad'] if chart else None,photo.size,result['counts'])['blocks']
            if guess!=rec['guess'][name]:
                raise ValueError(f'{key} / {name} changed during rendering: {guess}')
            clean,grid,_ = _straight_photo(chart,result) if chart else (None,None,None)
            if name in ('current','self_repeat'):
                cells.append((clean,'Usable texture pixels' if name=='current' else 'Crop selected by this trial'))
            if name=='coplanar':
                donor_id = result.get('evidence',{}).get('transfer',{}).get('donor')
                donor = next((p for p in parts if p['id']==donor_id),None)
                view = overlay(photo,(0,0,*photo.size),[(mask,(26,218,205),.8),(support(donor,photo.size),(226,95,246),.85)]) if donor else None
                cells.append((view,'Magenta donor / teal recipient' if donor else 'No transfer accepted'))
            cells.append((grid,'Count '+size_text(guess)))
        cells.append((overlay(photo,box,[(truth,(253,180,45),.67)]),'Recorded size '+size_text(rec['actual'])))
        x=left+col*w
        label(draw,(x,top-32),layout.title(),22,bold=True)
        for i,(img,caption) in enumerate(cells):
            y=top+i*h
            if img is not None:
                _picture(page,img,x+2,y+4,w-4,h-42,nearest=i>0)
            label(draw,(x+4,y+h-30),caption,15)
        y=top+(len(stages)-1)*h+20
        for j,name in enumerate(METHODS):
            g=rec['guess'][name]
            verdict='correct' if g==rec['actual'] else 'unresolved' if None in g else 'wrong'
            title={'current':'Default','self_repeat':'Repeats alone','phase':'With atlas','coplanar':'Same-plane'}[name]
            label(draw,(x+4,y+40*j),f'{title}: {verdict}',18)
        index[layout]=dict(key=key,sha256=frame['sha256'],proposal=part['id'],reference=rec['reference'],
                           actual=rec['actual'],guess=rec['guess'])
    page.save(out/'comparison.jpg',quality=92,subsampling=0)
    save(out/'examples.json',index)


@image_only()
def run(phase,coplanar,out,phase_key,coplanar_key):
    out.mkdir(parents=True,exist_ok=False)
    rows = join((phase,phase_key),(coplanar,coplanar_key))
    scores = {group:summarize([r for r in rows if group in ('all',r['cohort'],r['layout'])],METHODS)
              for group in dict.fromkeys(['all',*[r['cohort'] for r in rows],*[r['layout'] for r in rows]])}
    save(out/'scores.json',scores)
    save(out/'cases.json',rows)
    render(rows,out)
    save(out/'manifest.json',dict(code=revision(),run_id=out.name,version='26.3',
         source_manifests={p.name:digest(p/'manifest.json') for p in (BASE,phase,coplanar)},
         frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in rows],
         artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--phase',type=Path,required=True)
    cli.add_argument('--coplanar',type=Path,required=True)
    cli.add_argument('--phase-key',default='phase')
    cli.add_argument('--coplanar-key',default='coplanar')
    cli.add_argument('--out',type=Path,required=True)
    a=cli.parse_args()
    run(a.phase,a.coplanar,a.out,a.phase_key,a.coplanar_key)
