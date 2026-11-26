"""Build the comparison sheet and paired summary for the three seam methods."""
import argparse
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from benchmark.block_seam_round import MODELS, POOL
from benchmark.camera_metrics import project_visible
from benchmark.camera_round import digest, read, save
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.grid_chooser_sheet import reference_photo, size_text
from benchmark.mask_blocks import decoded_pack, SETTINGS
from benchmark.normal_round import groups
from benchmark.surface_pipeline_sheet import _picture, _straight_photo, PAPER, MUTED
from benchmark.surface_reference import recorded
from benchmark.surface_sheet import text
from benchmark.wall_join_test import checked,read_gzip
from engine.camera_grid import predict
from engine.strip_neighbor import clean_surfaces
from engine.surface_perimeter import context_crop, draw_outline, with_outline
from engine.surface_sample import sample_surface


def collect(runs):
    summaries,examples = {},defaultdict(dict)
    labels = {f['key']:f for f in read(POOL/'targets.json')}
    for method,path in runs.items():
        manifest = read(path/'manifest.json')
        if manifest['method']!=method or manifest['frame_manifest_sha256']!=digest(POOL/'manifest.json'):
            raise ValueError('Mismatched candidate or suite')
        for name,sha in manifest['artifacts'].items():
            checked(path/name,sha)
        placement = path/'placement'
        info = read(placement/'manifest.json')
        for name,sha in info['artifacts'].items():
            checked(placement/name,sha)
        dims, scores = read(path/'dimensions.json'), read(placement/'scores.json')
        table = {}
        for scope in ('all','development','regression'):
            rec = scores[scope]['ranker']
            grid = rec['grids']
            table[scope] = dict(**dims[scope],aligned=grid['aligned'],proposed=grid['proposed'],
                failed_alignment=grid['proposed']-grid['aligned'],unpaired_footprints=grid['proposed']-grid['iou_matches'],
                aligned_surfaces=grid['surfaces_with_aligned_face'],**rec['coordinates'])
        summaries[method] = table
        with ZipFile(placement/'geometry-packs.zip') as packs:
            for f,p in zip(read(path/'readings.json'),read(placement/'cases.json'),strict=True):
                if f['sha256']!=p['sha256'] or f['sha256']!=labels[f['key']]['sha256']:
                    raise ValueError('Mismatched image')
                sets = [decoded_pack(BytesIO(packs.read(p['exports'][k]['pack']))) for k in ('baseline','ranker')]
                for scope in ('all',f['cohort']):
                    s = table[scope]
                    s['changed_packs'] = s.get('changed_packs',0)+int(sets[0]!=sets[1])
                for surface,grade,target in zip(f['surfaces'],p['surfaces'],labels[f['key']]['surfaces'],strict=True):
                    if surface['id']!=grade['id'] or grade['reference_id']!=target['reference_id']:
                        raise ValueError('Mismatched surface')
                    if surface['quad'] is None or target['expected'] is None:
                        continue
                    key = (f['key'],surface['id'])
                    rows = examples[f['layout']].setdefault(key,dict(reference=target['reference_id'],expected=target['expected']))
                    rows['baseline'] = (surface['baseline'],grade['baseline'])
                    rows[method] = (surface['candidates'][0],grade['ranker'])
    return summaries,examples


def pick(views):
    def difference(key,metric):
        row = views[key]
        changes = [metric(row[m][1])-metric(row['baseline'][1]) for m in MODELS]
        size_change = any(row[m][0]['blocks']!=row['baseline'][0]['blocks'] for m in MODELS)
        return max(changes),size_change, row['baseline'][1]['proposed']
    gain = max(views,key=lambda k:difference(k,lambda g:g['aligned']))
    other = [k for k in views if k[0]!=gain[0]]
    cost = max(other,key=lambda k:difference(k,lambda g:g['proposed']-g['aligned']))
    return gain,cost


def export_view(image, data):
    cells = decoded_pack(BytesIO(data))
    view = image.convert('L').convert('RGB')
    if cells:
        P = np.asarray(predict(image,'26.3',SETTINGS['camera'])['P'])
        xy = project_visible(P,np.array(sorted(cells))+.5)
        draw = ImageDraw.Draw(view)
        for x,y in xy[np.isfinite(xy).all(1)]:
            draw.ellipse((x-4,y-4,x+4,y+4),fill='#ffcc42')
    return view,len(cells)


def render(runs,out):
    table,examples = collect(runs)
    out.mkdir(parents=True,exist_ok=True)
    save(out/'summary.json',table)
    cohorts,_,_ = groups()
    captures = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    raw = {r['key']:r for r in read(checked(BASE/'cases.json',source['artifacts']['cases.json']))}
    names = ['Screenshot','Detected surface','Straightened crop','Current counter','Outline edges',
             'Learned seams','Atlas match','Visible answer','Atlas export']
    left,col,head,sub = 181,262,124,172
    page = Image.new('RGB',(left+len(examples)*col+8,head+len(names)*(2*sub+8)),PAPER)
    draw = ImageDraw.Draw(page)
    text(draw,18,12,'Which method locates visible block seams?',32,True)
    text(draw,18,53,'Two examples per building. Gold marks measured seams. Crop captions give aligned / proposed faces. Green marks recorded visible boundaries.',18)
    for i,name in enumerate(names):
        y = head+i*(2*sub+8)
        draw.line((10,y,page.width-10,y),fill='#c7d3cf',width=2)
        text(draw,12,y+10,name,19,True)
    selected = []
    with ZipFile(runs['template']/'placement/geometry-packs.zip') as packs:
        for j,(layout,views) in enumerate(sorted(examples.items())):
            x = left+j*col
            text(draw,x,88,layout.title(),22,True)
            for i,key in enumerate(pick(views)):
                frame,pid = key
                info,r,c = views[key],raw[frame],captures[frame]
                with Image.open(checked(c.path('image'),r['sha256'])) as src:
                    image = src.convert('RGB')
                upstream = read_gzip(checked(BASE/r['predictions'],source['artifacts'][r['predictions']]))['gap']
                parts,_ = clean_surfaces(image,upstream)
                part = next(p for p in parts if p['id']==pid)
                marked = image.copy()
                draw_outline(marked,with_outline(part),width=3)
                _,box,_ = context_crop(image,part,margin_px=16)
                sample = sample_surface(image,part)
                panels = [(image,' '.join(frame.split('-')[-3:])),(marked.crop(box),pid),
                          (_straight_photo(sample,info['baseline'][0])[0],'Visible RGB')]
                for name in ('baseline','boundaries','learned','template'):
                    reading,g = info[name]
                    caption = f"{size_text(reading['blocks'])}     {g['aligned']} / {g['proposed']} faces"
                    panels.append((_straight_photo(sample,reading)[1],caption))
                _,ref = recorded(c)
                planes = reference_planes(ref,read(c.path('labels'))['faces'])
                truth = reference_photo(image,ref,planes,info['reference'])
                panels.append((truth.crop(box),size_text(info['expected'])+' visible extent'))
                projected,n = export_view(image,packs.read(frame+'/ranker.zip'))
                panels.append((projected,f'{n} exported block centers'))
                for stage,(photo,caption) in enumerate(panels):
                    y = head+stage*(2*sub+8)+i*sub+7
                    _picture(page,photo,x,y,col-9,139)
                    text(draw,x,y+142,caption,15,fill=MUTED)
                selected.append(dict(layout=layout,frame=frame,proposal=pid,reference=info['reference']))
    page.save(out/'comparison.jpg',quality=91)
    save(out/'sheet-views.json',selected)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for method in MODELS:
        parser.add_argument('--'+method,type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    render({m:getattr(args,m) for m in MODELS},args.out)
