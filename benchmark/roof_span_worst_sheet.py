"""Render whole-photo stage overlays and an enlarged map for each ranked structure."""
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw

from benchmark import framed_depth_eval as framed
from benchmark import roof_span_worst_stages as stages
from benchmark.framed_depth_sheet import draw_footprint
from benchmark.nearby_worst_views import COLORS
from benchmark.surface_pipeline_sheet import _picture, MAT, PAPER, INK, MUTED
from benchmark.surface_sheet import text
from benchmark.texture_camera_eval import STABLE, save
from engine import atlas_count, surface_perimeter
from training.visible_labels import visible_labels


LABELS = ('Source screenshot','Surface Finder','Surface crop','Surface Straightener','Block Counter',
          'Grid on source photo','Material Matcher','Build on source photo','Visible reference',
          'Placement errors','Footprint from above')


def face_map(photo, pose, positions, color):
    if not positions:
        return photo.copy(),None
    material = dict(block_id='smooth_stone',state='Block{minecraft:smooth_stone}')
    labels,ids,_ = visible_labels(pose,dict.fromkeys(positions,material),(0,0,0))
    lookup = np.zeros((len(labels['faces'])+1,3))
    for face in labels['faces']:
        p = tuple(face['xyz_world'])
        axis = int(np.argmax(np.abs(face['normal_world'])))
        lookup[face['id']] = np.asarray(color(p)) * (.85,1.,.74)[axis]
    fg = ids!=0
    data = np.array(photo)
    data[fg] = .68*lookup[ids[fg]]+.32*data[fg]
    edge = np.zeros_like(fg)
    edge[1:] |= ids[1:]!=ids[:-1]
    edge[:,1:] |= ids[:,1:]!=ids[:,:-1]
    data[edge&fg] = data[edge&fg]*.47
    return Image.fromarray(data),Image.fromarray(fg).getbbox()


def detail(image, box):
    if not box:
        return None
    x,y,right,bottom = box
    margin=max(12,round(max(right-x,bottom-y)*.06))
    return image.crop((max(0,x-margin),max(0,y-margin),min(image.width,right+margin),min(image.height,bottom+margin)))


def prepare(run, info, frame_tuple, atlas):
    rep = info['representative']
    frame,prior,row = frame_tuple
    image,parts,counts,camera,pose,labels,ids,truth,raw,scored,states = stages.read_view(run,rep['split'],frame,prior,row)
    missing = truth-scored
    if row['id'] in framed.roof_specs():
        height = framed.roof_specs()[row['id']]['roof_y']
        missing = {p for p in missing if p[1] == height} or missing
    results,selected,totals = stages.surface_evidence(image,parts,counts,labels,ids,missing,atlas)
    part = selected['part']
    record = selected['record']
    outline = image.copy()
    for p in parts:
        surface_perimeter.draw_outline(outline,p,'#ff923b',3)
    surface_perimeter.draw_outline(outline,part,'#40ffff',4)
    crop = surface_perimeter.context_crop(image,part,12)
    source_grid = stages.mark_source_grid(image,results,selected)
    clean,grid,pair,material = stages.material_pair(selected,atlas)
    if material.get('quad') is not None:
        location = np.asarray(material['quad']).mean(axis=0)
        u,v = np.floor(location).astype(int)
        owner = int(ids[v,u])
        block_id = labels['faces'][owner-1]['block'] if owner else None
        material['recorded_id_at_tile_center'] = labels['blocks'][block_id-1]['block_id'] if block_id else None
    restored,raw_box = face_map(image,pose,raw,lambda _: (64,225,213))
    observed,truth_box = face_map(image,pose,truth,lambda _: (182,165,254))
    errors,error_box = face_map(image,pose,truth|scored,lambda p: COLORS[0 if p in truth and p in scored else 1 if p in scored else 2])
    xz = [(x,z) for x,y,z in truth|scored]
    left,top = np.min(xz,axis=0)
    right,bottom = np.max(xz,axis=0)
    overhead = Image.new('RGB',(396,256),'#24343a')
    draw_footprint(overhead,scored,truth,(left,top,right,bottom),0,0)
    dims = record['dimensions']['blocks']
    dimension = ' × '.join('?' if v is None else str(v) for v in dims)
    phase = ['known' if v is not None else 'unknown' for v in record['phase']]
    geom = row['relative_geometry']
    roof = row.get('roof')
    roof_text = ''
    if roof:
        sp = roof['spacing']
        roof_text = f"Roof anchored {sp['anchored_roof_cells']}/{sp['visible_roof_cells']}. Opening {sp['observed_gap_width']}/{sp['target_gap_width']} blocks. {roof['filled_clear_rays']}/{roof['clear_rays']} sightlines blocked."
    def label(value):
        return value.removeprefix('minecraft:').replace('_',' ') if value else 'unresolved'
    evidence = dict(id=rep['id'],layout=info['name'],rank=info['rank'],shape_sha256=info['shape_sha256'],
        frame_hash=rep['rgb_sha256'],pack_hash=rep['pack_sha256'],counts=totals,selected=part['id'],
        selected_truth_surface=selected['matched_surface'],selected_truth_iou=selected['best_iou'],
        native_context=crop[1] if crop else None,method=record['method'],frozen_dimensions=dims,
        chart_quad=record.get('quad'),chart_counts=record['counts'],phase=record['phase'],
        selected_complete_tiles=len(selected['tiles']),material_replay=material,exported_materials=states,
        source_registration_available=camera.get('P') is not None,score_translation=row['translation'],
        roof=roof,relative_geometry=geom,span=row['span'])
    relevant = f"{info['f1']:.2%} pooled across {info['frames']} fixed view{'s' if info['frames']!=1 else ''}."
    stage_rows = [
        (image,None,f"Worst view {rep['f1']:.2%} F1. {relevant}"),
        (outline,detail(outline,truth_box),f"{totals['matched_at_iou_half']}/{totals['visible']} visible surfaces matched at 0.5 IoU. {totals['predicted']} proposals. Cyan traces the selected surface."),
        (outline.crop(crop[1]) if crop else None,None,f"Original pixels with a 12 px margin around {part['id']}."),
        (clean,None,f"{totals['charts']}/{totals['predicted']} surfaces have a measured chart. Dark pixels lie outside the selected mask."),
        (grid,None,f"Across × down: {dimension} blocks. Grid origins {phase[0]} / {phase[1]}. {totals['two_dimension_counts']}/{totals['predicted']} surfaces have two measured dimensions."),
        (source_grid,detail(source_grid,truth_box),f"Located grid lines mapped to the source. {totals['two_grid_origins']}/{totals['predicted']} surfaces have both grid origins. Yellow highlights the selected grid."),
        (pair,None,(f"Photo / atlas: {label(material.get('measured_block_id'))}. Recorded: {label(material.get('recorded_id_at_tile_center'))}. Tile from {material['origin_source']}." if pair else material['reason'])),
        (restored,detail(restored,raw_box),f"{geom['placed']} exported blocks. Teal marks decoded geometry in the registered camera frame."),
        (observed,detail(observed,truth_box),f"{geom['target']} source blocks had a visible face. Purple marks the scored reference."),
        (errors,detail(errors,error_box),f"{geom['correct']} correct (green), {geom['extras']} extra (red), {geom['missed']} missing (blue). One shared shift {tuple(row['translation'])}."),
        (overhead,None,f"Highest block per column. X/Z spans {row['span']['after'][0]}/{row['span']['after'][2]}; visible target {row['span']['target'][0]}/{row['span']['target'][2]}. {roof_text}")]
    return stage_rows,evidence


def panel(image, box, content, label=None):
    x,y,w,h = box
    picture,inset,caption = content
    pen = ImageDraw.Draw(image)
    if label:
        text(pen,x,y,label,23,True,INK)
        y += 37
        h -= 37
    image_h = h-100
    first = round(image_h*.64) if inset else image_h
    if picture is not None:
        _picture(image,picture,x,y,w,first)
    else:
        pen.rectangle((x,y,x+w,y+first),fill=MAT)
        text(pen,x+15,y+first//2,'Unresolved on this surface',19,False,MUTED)
    if inset:
        _picture(image,inset,x,y+first+5,w,image_h-first-5)
    for i,line in enumerate(wrap(caption, width=max(29,int(w/9.8)))):
        text(pen,x+2,y+image_h+11+i*23,line,17,False,MUTED)


def render(run, out, ranking, frames):
    atlas = atlas_count.load_atlas(STABLE/'runs/26.3/atlas','26.3')
    layouts = ranking['layouts']
    width,lead,slot,head = 465,233,480,156
    page = Image.new('RGB',(lead+len(layouts)*slot+10,head+len(LABELS)*491+15),PAPER)
    draw = ImageDraw.Draw(page)
    text(draw,15,13,'Ten lowest-scoring structures',39,True)
    text(draw,16,65,'Fixed evaluation • rank by pooled visible-block F1 after one translation • each column shows its lowest-scoring full-frame view',21,False,MUTED)
    for i,stage in enumerate(LABELS):
        y = head+i*491
        draw.line((14,y,page.width-16,y),fill='#bac8c6',width=2)
        text(draw,14,y+15,stage,19,True)
    traces=[]
    (out/'overlays').mkdir(exist_ok=True)
    for j,layout in enumerate(layouts):
        view = layout['representative']
        panels,evidence = prepare(run,layout,frames[view['split'],view['id']],atlas)
        for stage,idx in [('source',0),('outlines',1),('grid',5),('build',7),('truth',8),('errors',9)]:
            panels[idx][0].save(out/'overlays'/f'{j+1:02}-{stage}.jpg',quality=89,subsampling=0,optimize=True)
        traces.append(evidence)
        x = lead+j*slot
        text(draw,x,99,f"{j+1}. {layout['name']}  ·  {layout['f1']:.2%}",25,True)
        text(draw,x,131,f"Worst view {view['f1']:.2%}  ·  {layout['frames']} fixed view(s)",16,False,MUTED)
        individual = Image.new('RGB',(1450,142+6*625),PAPER)
        pen = ImageDraw.Draw(individual)
        text(pen,21,14,f"{j+1}. {layout['name']}   •   {layout['f1']:.2%} pooled",36,True)
        text(pen,22,71,f"Representative image: {view['id']}   •   {view['f1']:.2%} F1",21,False,MUTED)
        for i,(stage,content) in enumerate(zip(LABELS,panels)):
            panel(page,(x,head+i*491+10,width,469),content)
            col,row=i%2,i//2
            y=128+row*625
            px=20+col*720
            pen.line((px,y,px+694,y),fill='#bac8c6',width=2)
            panel(individual,(px,y+12,691,598),content,stage)
        individual.save(out/layout['sheet'],quality=91,subsampling=0,optimize=True)
    page.save(out/'comparison.jpg',quality=92,subsampling=0,optimize=True)
    save(out/'stage-traces.json',dict(cases_sha256=ranking['cases_sha256'],stages=LABELS,atlas=atlas.provenance,views=traces))
    from benchmark.roof_span_worst_text import write_report
    write_report(out,ranking,traces,run)
