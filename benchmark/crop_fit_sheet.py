"""Portrait comparison on eight previously designated crop-review surfaces."""
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read, save
from benchmark.camera_sheet import label
from benchmark.crop_review_sheet import clipped_box
from benchmark.ledge_join_sheet import picture
from benchmark.normal_round import groups
from benchmark.surface_detection_sheet import paint
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from engine import masked_count, surface_perimeter
from engine.surface_sample import sample_surface


EXAMPLES = ['Shopfront B', 'Wing A', 'Courtyard B', 'Portico A', 'Workshop C',
            'Crossing A', 'Galleries A', 'Registry C']


def size_text(pair):
    return ' × '.join('?' if value is None else str(value) for value in (pair or [None, None]))


def render(out):
    from benchmark.crop_full_fit import sample as full
    from benchmark.crop_guarded_fit import sample as guarded

    source = ROOT / 'benchmark/reviews/sky-gap-01'
    rows = {entry['key']: entry for entry in read(source / 'cases.json')}
    scores = {entry['key']: entry for entry in read(out / 'cases.json')}
    examples = {entry['id']: entry for entry in read(ROOT / 'benchmark/reviews/crop-review-02/index.json')['examples']}
    captures = {case.floor+'/'+case.id: case for group in groups()[0].values() for case in group}
    modes = [('current', 'Current crop', sample_surface), ('full', 'Full outline', full),
             ('guarded', 'Guarded expansion', guarded)]
    stages = ['Source image', 'Recorded surface', 'Surface Finder'] + [
        stage for _, name, _ in modes for stage in (name, 'Face-on crop', 'Block Counter')]
    width, height = 346, 270
    left, top = 227, 167
    page = Image.new('RGB', (left+len(EXAMPLES)*width, top+len(stages)*height+10), '#f7f6ef')
    pen = ImageDraw.Draw(page)
    label(pen, (15,12), 'Does a larger crop help count blocks?', 38, bold=True)
    label(pen, (15,67), 'Gold = recorded surface or predicted grid. Cyan = detected outline. Violet = sampling window. Coverage measures the fraction of the detected surface inside the window.', 21)
    label(pen, (15,100), 'Dark gray pixels are excluded from the count. A question mark means that the counter could not resolve that dimension.', 19)
    for pos, stage in enumerate(stages):
        y = top+pos*height
        pen.line((8,y,page.width-8,y),fill='#b2bfb9',width=2)
        label(pen,(12,y+15),stage,21,bold=True)
    views = []
    for col, name in enumerate(EXAMPLES):
        ex = examples[name]
        key, rid = ex['frame'], ex['reference']
        case = scores[key]
        if case['sha256'] != ex['sha256']:
            raise ValueError('Review capture changed: '+key)
        row = next(r for r in case['matched'] if r['reference_id']==rid)
        photo, _, native, grades = picture(source, rows[key], captures[key], ['gap'])
        g = grades['gap']
        part = g['parts'][g['lookup'][rid]['index']-1]
        if part['id'] != row['proposal']:
            raise ValueError('Mismatch in reviewed proposal')
        outlined = photo.copy()
        surface_perimeter.draw_outline(outlined, part, '#2de9d4', 2)
        detail = ex['detail_xyxy']
        corners = [p for m in row['methods'].values() for p in (m['quad'] or [])]
        box = clipped_box([detail[:2], detail[2:], *corners], photo.size, 3)
        location = photo.copy()
        ImageDraw.Draw(location).rectangle(box, outline='white', width=2)
        actual = size_text(row['expected'])
        cells = [(location.crop(ex['scene_xyxy']), key.split('/')[-1]),
                 (paint(photo,native,{rid:(255,194,73)}).crop(box), f'Actual {actual} blocks   ·   R{rid}'),
                 (outlined.crop(box), 'Same detected surface for all three crops')]
        for method, _, sampler in modes:
            result = row['methods'][method]
            chart = outlined.copy()
            quad = [tuple(p) for p in (result['quad'] or [])]
            if quad:
                ImageDraw.Draw(chart).line([*quad,quad[0]],fill='#e27cff',width=2)
            cells.append((chart.crop(box), f"{result['covered']/row['source_mask_pixels']:.1%} of detected surface"))
            sample = sampler(photo,part)
            clean, grid = (None,None) if sample is None else _straight_photo(sample,masked_count.count(sample))[:2]
            cells.extend([(clean,'Visible pixels used by counter'), (grid,f"Counter {size_text(result['guess'])}   |   Actual {actual}")])
        x = left+col*width
        label(pen,(x+3,top-35),name,24,bold=True)
        for i,(pic,caption) in enumerate(cells):
            y = top+i*height
            if pic is not None:
                _picture(page,pic,x+2,y+6,width-7,height-45)
            label(pen,(x+3,y+height-33),caption,15)
        views.append(dict(example=name, frame=key, sha256=case['sha256'], reference=rid,
                          detail_xyxy=box, actual=row['expected'], methods=row['methods']))
    page.save(out/'comparison.jpg',quality=91,subsampling=0)
    save(out/'sheet-views.json',views)
