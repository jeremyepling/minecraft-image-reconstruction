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


STAGES = ('Original screenshot', 'Surface Finder', 'Crop with context',
           'Surface Straightener', 'Block Counter', 'Material Matcher')
MARGIN, GUTTER, COL = 18, 14, 350
TOP, PIC_H = 80, 202
CAPTION_HEIGHTS = (0,30,30,30,76,54)
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


def _detail_box(size,center,side):
    width,height=size
    side=max(1,min(round(side),width,height))
    x=int(np.clip(round(center[0]-side/2),0,width-side))
    y=int(np.clip(round(center[1]-side/2),0,height-side))
    return (x,y,x+side,y+side)


def _column(example, atlas):
    layout, key = str(example['layout']), str(example['key'])
    part, image = example['part'], example['image'].convert('RGB')
    margin = int(example.get('margin',surface_perimeter.DEFAULT_CONTEXT_PX))
    version = atlas.provenance['version']
    box = tuple(map(int,example['box']))
    starts = [TOP]
    for caption_height in CAPTION_HEIGHTS:
        starts.append(starts[-1]+34+PIC_H+caption_height+20)
    page = Image.new('RGB',(2*MARGIN+COL,starts[-1]),PAPER)
    draw = ImageDraw.Draw(page)
    text(draw,MARGIN,12,layout.title(),32,True)
    text(draw,MARGIN,52,example.get('extra') or str(example.get('source',key)).split('/')[-1],17,fill=MUTED)
    for label,top in zip(STAGES,starts):
        draw.line((MARGIN,top-5,page.width-MARGIN,top-5),fill=BORDER)
        text(draw,MARGIN,top+4,label,23,True)
    y = [top+34 for top in starts[:-1]]
    _picture(page,image.crop(box),MARGIN,y[0],COL,PIC_H)
    final = surface_perimeter.with_outline(part)
    marked = image.copy()
    surface_perimeter.draw_outline(marked,final,CYAN,3)
    _picture(page,marked.crop(box),MARGIN,y[1],COL,PIC_H)
    rings = final['outline']['rings']
    outer = sum(r['kind']=='outer' for r in rings)
    holes = len(rings)-outer
    support = part['support']
    area = sum(b-a for _,a,b in support['runs'])
    status = dict(layout=layout,key=key,source_image=str(example.get('source',key)),proposal=part['id'],
                  finder=example.get('finder','mask-first'),source_size=list(image.size),
                  scene_xyxy=list(box),card_stage_y=starts,
                  support=dict(size=support['size'],pixels=area,outer_rings=outer,holes=holes),
                  reference_whole_surface_blocks=example.get('expected'))
    if example.get('extra'):
        status['extra'] = str(example['extra'])
    captions = [top+PIC_H+8 for top in y]
    text(draw,MARGIN,captions[1],f'{outer} outer  ·  {holes} enclosed openings',17,fill=MUTED)
    details=[None]*len(STAGES)
    if rings:
        points=np.vstack([ring['points'] for ring in rings])*np.array(image.size)/support['size']
        center=(np.asarray(box[:2])+box[2:])/2
        closest=points[np.argmin(np.sum((points-center)**2,axis=1))]
        inset=_detail_box(image.size,closest,.26*min(box[2]-box[0],box[3]-box[1]))
        details[0]=('Source detail',image.crop(inset))
        details[1]=('Same detail with outline',marked.crop(inset))
        status['detail_native_xyxy']=list(inset)
    context = surface_perimeter.context_crop(image,part,margin)
    if context is None:
        _empty(page,MARGIN,y[2],'No visible pixels')
        status['context'] = None
    else:
        raw,bounds,tight = context
        # Use the contiguous original RGB neighborhood. The mask rim alone is added for display.
        marked_crop = marked.crop(bounds)
        _picture(page,marked_crop,MARGIN,y[2],COL,PIC_H)
        text(draw,MARGIN,captions[2],f'{raw.width} × {raw.height} px   ·   {margin} px margin',17,fill=MUTED)
        status['context'] = dict(native_xyxy=list(bounds),tight_xyxy_local=list(tight),margin_px=margin)
        cut,_=surface_perimeter.masked_crop(image,part)
        matte=Image.new('RGB',cut.size,MASKED)
        matte.paste(cut,mask=cut.getchannel('A'))
        details[2]=('Owned pixels only',matte)

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
        _empty(page,MARGIN,y[3],'No straightening chart')
        _empty(page,MARGIN,y[4],'Grid unresolved')
        status['straightener'] = None
    else:
        clean,grid,lines = _straight_photo(sample,result,tile['cell'] if tile else None)
        _picture(page,clean,MARGIN,y[3],COL,PIC_H)
        _picture(page,grid,MARGIN,y[4],COL,PIC_H)
        mask = sample['mask']
        status['straightener'] = dict(quad=q,size=list(clean.size),valid_pixels=int(mask.sum()),
                                      valid_fraction=float(mask.mean()),chart_source=sample['chart_source'])
        text(draw,MARGIN,captions[3],f'{mask.mean():.0%} of straightened image sampled',17,fill=MUTED)
        ys,xs=np.nonzero(mask)
        if len(xs):
            center=np.array(clean.size)/2
            i=np.argmin((xs-center[0])**2+(ys-center[1])**2)
            inset=_detail_box(clean.size,(xs[i],ys[i]),.4*min(clean.size))
            details[3]=('Texture detail',clean.crop(inset))
            if any(lines):
                details[4]=('Grid detail',grid.crop(inset))
    for axis,word in enumerate(('Across','Down')):
        text(draw,MARGIN,captions[4]+23*axis,word+'   '+_axis_status(dims[axis],result['phase'][axis]),17)
    status['counter'] = dict(method=method,chart_spans_blocks=result['counts'],phase_blocks=result['phase'],
                             whole_surface_blocks=dims,visible_grid_lines_uv=lines)
    if example.get('expected') is not None:
        across,down = example['expected']
        text(draw,MARGIN,captions[4]+46,f'Recorded size   {across} × {down}',17,fill=MUTED)

    status['material'] = dict(method='rank_tile',complete_cells=ntiles,cell=None)
    if tile is None:
        _empty(page,MARGIN,y[5],'No complete block located')
        text(draw,MARGIN,captions[5],'Material unresolved',17,fill=MUTED)
    else:
        ranked = atlas_count.rank_tile(tile,atlas,version=version)
        texture = ranked['texture']
        cell_width = (COL-12)//2
        _picture(page,Image.fromarray(tile['rgb']),MARGIN,y[5],cell_width,PIC_H,nearest=True)
        if texture is not None:
            sprite = Image.fromarray(np.uint8(atlas.tiles[atlas.labels.index(texture)]))
            _picture(page,sprite,MARGIN+cell_width+12,y[5],cell_width,PIC_H,nearest=True)
        text(draw,MARGIN,captions[5],'Photo',17,fill=MUTED)
        text(draw,MARGIN+cell_width+12,captions[5],'Atlas',17,fill=MUTED)
        name = ranked['block_id']
        guess = 'Guess   '+name.removeprefix('minecraft:').replace('_',' ') if name else 'Material unresolved'
        text(draw,MARGIN,captions[5]+23,guess,18,True)
        status['material'].update(cell=tile['cell'],native_quad=tile['quad'],top_id=name,
                                  texture=texture,ranked_ids=ranked['ranked_ids'])
    return page,status,details


def build(examples,atlas):
    """Compose one portrait matrix. Additional views share their building's column."""
    buildings={}
    views=[]
    for example in examples:
        tile,record,details=_column(example,atlas)
        buildings.setdefault(record['layout'],[]).append((tile,record,details))
        views.append(record)
    if not buildings:
        raise ValueError('A comparison sheet requires at least one building')
    label_width=214
    width=label_width+len(buildings)*(COL+GUTTER)+MARGIN
    heading=156
    max_views=max(map(len,buildings.values()))
    heights=[]
    for caption in CAPTION_HEIGHTS:
        single=PIC_H+caption+48
        heights.append(max(single*max_views+GUTTER,single+184,(width-heading)//len(STAGES)+24))
    height=heading+sum(heights)+MARGIN
    page=Image.new('RGB',(width,height),PAPER)
    draw=ImageDraw.Draw(page)
    text(draw,MARGIN,14,'Reconstruction stages',37,True)
    text(draw,MARGIN,63,'Cyan marks a surface. Gold marks the block grid. Blue marks the sampled block.',18,fill=MUTED)
    stages=[]
    for i,label in enumerate(STAGES):
        top=heading+sum(heights[:i])
        stages.append(dict(name=label,y0=top,y1=top+heights[i]))
        draw.line((MARGIN,top,width-MARGIN,top),fill=BORDER,width=2)
        text(draw,MARGIN,top+22,label,21,True)
    for j,(layout,parts) in enumerate(buildings.items()):
        x=label_width+j*(COL+GUTTER)
        text(draw,x,109,layout.title(),27,True)
        for tile,record,details in parts:
            record['sheet_column_xyxy']=[x,97,x+COL,height-MARGIN]
            record['sheet_stage_xyxy']=[]
        for i,stage in enumerate(stages):
            top=stage['y0']
            slot_height=heights[i]//len(parts)
            for k,(tile,record,details) in enumerate(parts):
                start=top+k*slot_height+12
                if len(parts)>1:
                    text(draw,x,start,record.get('extra','Primary surface'),16,True,fill=MUTED)
                    start+=26
                card_y=record['card_stage_y'][i]+34
                cell=tile.crop((MARGIN,card_y,MARGIN+COL,card_y+PIC_H+CAPTION_HEIGHTS[i]))
                page.paste(cell,(x,start))
                record['sheet_stage_xyxy'].append([x,start,x+COL,start+cell.height])
                if len(parts)==1 and details[i] is not None:
                    title,photo=details[i]
                    y=start+cell.height+14
                    text(draw,x,y,title,16,fill=MUTED)
                    h=min(PIC_H,stage['y1']-y-40)
                    _picture(page,photo,x,y+23,COL,h,nearest=True)
            if len(parts)>1:
                for k in range(1,len(parts)):
                    y=top+k*slot_height
                    draw.line((x,y,x+COL,y),fill=BORDER)
    for v in views:
        del v['card_stage_y']
    records=dict(schema=2,version=atlas.provenance['version'],atlas=atlas.provenance,
                 columns=list(buildings),stages=stages,image='comparison.jpg',views=views)
    return page,records


def stage_sheet(out, examples, atlas):
    """Write the full-resolution matrix as a single zoomable JPEG."""
    directory=Path(out)
    directory.mkdir(parents=True,exist_ok=True)
    page,records=build(examples,atlas)
    page.save(directory/'comparison.jpg',quality=94,subsampling=0)
    save(directory/'pipeline-rows.json',records)
    return records
