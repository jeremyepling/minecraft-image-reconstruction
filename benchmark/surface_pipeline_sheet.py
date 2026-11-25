"""Show a single predicted surface through each stage of the exterior pipeline."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import save
from benchmark.surface_sheet import text
from engine import atlas_count, masked_count, surface_perimeter
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from engine.surface_tiles import visible_tiles


COLUMNS = ('Original screenshot', 'Surface Finder', 'Crop with context',
           'Surface Straightener', 'Block Counter', 'Material Matcher')
NOTES = ('Scene crop from the original frame', 'Cyan follows the visible perimeter', 'Neighbors around the selected surface',
         'Visible pixels turned face-on', 'Gold grid • blue sampled block', 'Photographed cell and top atlas match')
MARGIN, GUTTER, COL = 28, 14, 320
TOP, ROW, PIC_H = 154, 308, 184
PAPER, INK, MUTED, BORDER = '#f8f7f1', '#183135', '#4b6465', '#ccd4d0'
MAT, MASKED, CYAN, GRID = '#e6eae5', '#293233', '#49ffff', '#ffc943'


def _picture(page, photo, x, y, w, h, *, nearest=False):
    draw = ImageDraw.Draw(page)
    draw.rectangle((x,y,x+w-1,y+h-1),fill=MAT)
    scale = min(w/photo.width,h/photo.height)
    method = Image.Resampling.NEAREST if nearest or scale>1.25 else Image.Resampling.LANCZOS
    shown = photo.resize((max(1,round(photo.width*scale)),max(1,round(photo.height*scale))),method)
    page.paste(shown,(x+(w-shown.width)//2,y+(h-shown.height)//2))


def _empty(page, x, y, message):
    draw = ImageDraw.Draw(page)
    draw.rectangle((x,y,x+COL-1,y+PIC_H-1),fill=MAT)
    text(draw,x+14,y+PIC_H//2-10,message,17,fill=MUTED)


def _straight_photo(sample, result, sampled_cell=None):
    """Draw only measured axes, clipped to the same mask which admitted the sampled colors."""
    mask = sample['mask']
    rgb = np.asarray(sample['rgb']).copy()
    rgb[~mask] = (41,50,51)
    clean = Image.fromarray(rgb)
    height,width = mask.shape
    if sampled_cell is not None:
        # visible_tiles emits only complete supported cells and gives their grid indices.
        ns = np.asarray(result['counts'])
        origin = np.asarray(sampled_cell)+np.mod(result['phase'],1.)
        y,x = np.ogrid[:height,:width]
        u,v = (x+.5)/width*ns[0], (y+.5)/height*ns[1]
        cell = mask & (u>=origin[0]) & (u<origin[0]+1) & (v>=origin[1]) & (v<origin[1]+1)
        rgb[cell] = .57*rgb[cell]+.43*np.array([64,170,255])
    lines = [[],[]]
    drawn = np.zeros(mask.shape,bool)
    for axis,(n,phase,extent) in enumerate(zip(result['counts'],result['phase'],(width,height))):
        if phase is None or n is None or n<=0:
            continue
        uv = np.arange(phase%1.,n+1.e-8,1.)/n
        coords = np.clip(np.rint(uv*extent-.5).astype(int),0,extent-1)
        for at,position in zip(coords,uv):
            band = slice(at,min(at+max(1,round(extent/COL)),extent))
            stripe = np.s_[:,band] if axis==0 else np.s_[band,:]
            if mask[stripe].any():
                drawn[stripe] = True
                lines[axis].append(float(position))
    rgb[drawn & mask] = (255,201,67)
    return clean,Image.fromarray(rgb),lines


def _axis_status(n,origin):
    if n is None:
        return 'count unknown'
    return f'{n} blocks  ·  '+('grid located' if origin is not None else 'grid origin unknown')


def _row(page, row_y, example, atlas):
    layout, key = str(example['layout']), str(example['key'])
    part, image = example['part'], example['image'].convert('RGB')
    margin = int(example.get('margin',surface_perimeter.DEFAULT_CONTEXT_PX))
    version = atlas.provenance['version']
    box = tuple(map(int,example['box']))
    x = [MARGIN+j*(COL+GUTTER) for j in range(len(COLUMNS))]
    y = row_y+34
    draw = ImageDraw.Draw(page)
    title = layout.title()
    if example.get('extra'):
        title += '   ·   '+str(example['extra'])
    text(draw,x[0],row_y,title,22,True)
    text(draw,x[3],row_y+4,key+'   ·   '+str(part['id']),16,fill=MUTED)
    _picture(page,image.crop(box),x[0],y,COL,PIC_H)
    final = surface_perimeter.with_outline(part)
    marked = image.copy()
    surface_perimeter.draw_outline(marked,final,CYAN,3)
    _picture(page,marked.crop(box),x[1],y,COL,PIC_H)
    rings = final['outline']['rings']
    outer = sum(r['kind']=='outer' for r in rings)
    holes = len(rings)-outer
    support = part['support']
    area = sum(b-a for _,a,b in support['runs'])
    status = dict(layout=layout,key=key,source_image=str(example.get('source',key)),proposal=part['id'],
                  finder=example.get('finder','mask-first'),source_size=list(image.size),
                  scene_xyxy=list(box),sheet_row_xyxy=[0,row_y,page.width,row_y+ROW],
                  support=dict(size=support['size'],pixels=area,outer_rings=outer,holes=holes),
                  reference_whole_surface_blocks=example.get('expected'))
    if example.get('extra'):
        status['extra'] = str(example['extra'])
    captions_y = y+PIC_H+11
    text(draw,x[1],captions_y,f'{outer} outer • {holes} enclosed openings',16,fill=MUTED)
    context = surface_perimeter.context_crop(image,part,margin)
    if context is None:
        _empty(page,x[2],y,'No visible pixels')
        status['context'] = None
    else:
        raw,bounds,tight = context
        # Use the contiguous original RGB neighborhood. The mask rim alone is added for display.
        marked_crop = marked.crop(bounds)
        _picture(page,marked_crop,x[2],y,COL,PIC_H)
        text(draw,x[2],captions_y,f'{raw.width} × {raw.height} px   ·   {margin} px margin',16,fill=MUTED)
        status['context'] = dict(native_xyxy=list(bounds),tight_xyxy_local=list(tight),margin_px=margin)

    sample = sample_surface(image,part)
    method = example.get('counter','masked-repeat')
    if method=='atlas-repeat':
        result = atlas_count.count(sample,atlas,version=version) if sample is not None else dict(counts=[None,None],phase=[None,None])
    elif method=='legacy-repeat':
        from engine.legacy_count import count
        result=count(sample,version=version)
    elif method=='masked-repeat':
        result=masked_count.count(sample,version=version)
    else:
        raise ValueError(f'Unknown Block Counter method {method}')
    q = None if sample is None else sample['quad']
    dims = dimensions(part,q,image.size,result['counts'])['blocks']
    ntiles = 0
    tile = None
    for candidate in visible_tiles(sample,result):
        ntiles += 1
        if tile is None:
            tile = candidate
    lines = [[],[]]
    if sample is None:
        _empty(page,x[3],y,'No straightening chart')
        _empty(page,x[4],y,'Grid unresolved')
        status['straightener'] = None
    else:
        clean,grid,lines = _straight_photo(sample,result,tile['cell'] if tile else None)
        _picture(page,clean,x[3],y,COL,PIC_H)
        _picture(page,grid,x[4],y,COL,PIC_H)
        mask = sample['mask']
        status['straightener'] = dict(quad=q,size=list(clean.size),valid_pixels=int(mask.sum()),
                                      valid_fraction=float(mask.mean()),chart_source=sample['chart_source'])
        text(draw,x[3],captions_y,f'{mask.mean():.0%} of straightened image sampled',16,fill=MUTED)
    for axis,word in enumerate(('Across','Down')):
        text(draw,x[4],captions_y+22*axis,word+'   '+_axis_status(dims[axis],result['phase'][axis]),16)
    status['counter'] = dict(method=method,chart_spans_blocks=result['counts'],phase_blocks=result['phase'],
                             whole_surface_blocks=dims,visible_grid_lines_uv=lines)
    if example.get('expected') is not None:
        across,down = example['expected']
        text(draw,x[4],captions_y+44,f'Recorded size   {across} × {down}',15,fill=MUTED)

    status['material'] = dict(method='rank_tile',complete_cells=ntiles,cell=None)
    if tile is None:
        _empty(page,x[5],y,'No complete block located')
        text(draw,x[5],captions_y,'Material unresolved',16,fill=MUTED)
    else:
        ranked = atlas_count.rank_tile(tile,atlas,version=version)
        texture = ranked['texture']
        cell_width = (COL-12)//2
        _picture(page,Image.fromarray(tile['rgb']),x[5],y,cell_width,PIC_H,nearest=True)
        if texture is not None:
            sprite = Image.fromarray(np.uint8(atlas.tiles[atlas.labels.index(texture)]))
            _picture(page,sprite,x[5]+cell_width+12,y,cell_width,PIC_H,nearest=True)
        text(draw,x[5],captions_y,'Photo',16,fill=MUTED)
        text(draw,x[5]+cell_width+12,captions_y,'Atlas',16,fill=MUTED)
        name = ranked['block_id']
        guess = 'Guess   '+name.removeprefix('minecraft:').replace('_',' ') if name else 'Material unresolved'
        text(draw,x[5],captions_y+23,guess,17,True)
        status['material'].update(cell=tile['cell'],native_quad=tile['quad'],top_id=name,
                                  texture=texture,ranked_ids=ranked['ranked_ids'])
    draw.line((MARGIN,row_y+ROW-12,page.width-MARGIN,row_y+ROW-12),fill=BORDER)
    return status


def build(examples, atlas):
    """Return (RGB sheet, JSON-serializable row manifest). All examples are explicit, ordered inputs.

    Each needs layout, key, source image (PIL RGB), scene box (native xyxy), and a final predicted
    part. Optional margin is in source pixels. expected is an evaluation-only [across,down] grade,
    extra a display label, source an image filename, and finder the actual detector method.
    """
    examples = list(examples)
    width = 2*MARGIN+len(COLUMNS)*COL+(len(COLUMNS)-1)*GUTTER
    page = Image.new('RGB',(width,TOP+ROW*len(examples)),PAPER)
    draw = ImageDraw.Draw(page)
    text(draw,MARGIN,19,'From surface outline to individual blocks',34,True)
    text(draw,MARGIN,69,'Cyan traces the selected surface. Dark areas are excluded when sampling. Each row follows one surface from one screenshot.',19,fill=MUTED)
    for j,(name,note) in enumerate(zip(COLUMNS,NOTES)):
        x = MARGIN+j*(COL+GUTTER)
        text(draw,x,105,name,22,True)
        text(draw,x,134,note,15,fill=MUTED)
    rows = [_row(page,TOP+i*ROW,e,atlas) for i,e in enumerate(examples)]
    return page,dict(version=atlas.provenance['version'],atlas=atlas.provenance,
                     columns=list(COLUMNS),image='comparison.jpg',rows=rows)


def stage_sheet(out, examples, atlas):
    """Write comparison.jpg and pipeline-rows.json to the round directory."""
    page,records = build(examples,atlas)
    directory = Path(out)
    directory.mkdir(parents=True,exist_ok=True)
    page.save(directory/'comparison.jpg',quality=94,subsampling=0)
    save(directory/'pipeline-rows.json',records)
    return records
