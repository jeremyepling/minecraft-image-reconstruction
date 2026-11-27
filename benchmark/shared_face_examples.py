"""Map weak source images and count missed cubes that actually show multiple faces."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import facade_inset_round as suite, roof_span_worst_sheet as sheet
from benchmark.inset_default_verification import local_release
from benchmark.mask_blocks import registered
from benchmark.shared_face_error_budget import current_pack
from benchmark.strip_visibility_audit import block_footprints
from benchmark.roof_span_worst import view_f1
from benchmark.surface_pipeline_sheet import PAPER, MUTED
from benchmark.surface_sheet import text
from benchmark.texture_camera_eval import digest, read, save
from engine import atlas_count

ROOT = suite.ROOT/'benchmark/reviews'
OUT = ROOT/'shared-face-round-01'
OLD_LAYOUTS = ROOT/'facade-placement-01/worst/ranking.json'


def grade_faces(job):
    """Scoring-side diagnostic only. A second recorded face need not be measurable from RGB."""
    split, frame, _, _ = job
    cells, _, shift, metrics = current_pack(job)
    f = suite.framed
    truth, _, transform, _ = f.references(split, frame, f.NATIVE_INPUTS.get(split),
        f.TRAINING_DATA, f.DEPTH, dict(native=f.native.BUILDS,old=f.transfer.BUILDS,fresh=f.transfer.BUILDS))
    placed = f.spacing.shift(registered(cells, transform), shift)
    missing = truth - placed
    assert (len(missing),len(placed-truth)) == (metrics['missed'],metrics['extras'])
    labels, _ = suite.visible_map(split,frame,metadata=True)
    faces = defaultdict(list)
    for face in labels['faces']:
        faces[tuple(face['xyz_world'])].append(face['visible_pixels'])
    assert set(faces)==truth
    distribution = Counter(len(faces[p]) for p in missing)
    # This illustrative cutoff is an upper bound on the number of eligible
    # targets, even before the full face, common edge or grid can be inferred.
    broad_and_slender = 0
    for pos in missing:
        areas = sorted(faces[pos], reverse=True)
        if areas[0] >= 64 and any(4 <= a <= areas[0]/3 for a in areas[1:]):
            broad_and_slender += 1
    return dict(id=frame['id'], split=split, rgb_sha256=frame['files']['rgb.png'],
        pose_sha256=frame['files']['camera.json'], correct=metrics['correct'], missed=len(missing),
        extras=metrics['extras'],missed_by_visible_faces={str(i):distribution[i] for i in (1,2,3)},
        broad_and_smaller=broad_and_slender)


def draw_examples(jobs, rows):
    by_key = {r['id']:r for r in rows}
    layouts = read(OLD_LAYOUTS)['all_layouts']
    names = {k:l for l in layouts for k in l['frame_ids']}
    ordered = sorted(rows,key=lambda r:(2*r['correct']/(2*r['correct']+r['missed']+r['extras']),r['id']))
    selected = {}
    for row in ordered:
        layout = names[row['id']]
        if layout['shape_sha256'] not in selected:
            selected[layout['shape_sha256']]=row['id']
        if len(selected)==5:
            break
    jobs_by_key = {j[1]['id']:j for j in jobs}
    atlas = atlas_count.load_atlas(suite.ATLAS,'26.3')
    frozen = {r['id']:r for r in read(ROOT/'facade-inset-round-01/combined.json')['rows']}
    traces = []
    overlays = OUT/"overlays"
    overlays.mkdir(exist_ok=True)
    stages = [(0,'Source image'),(1,'Surface Finder'),(2,'Surface crop'),(3,'Straightened crop'),(4,'Block Counter'),
              (5,'Grid on source image'),(6,'Material sample'),(7,'Projected build'),(8,'Visible reference'),(9,'Placement errors')]
    w, h, top, left = 480, 402, 141, 205
    page = Image.new('RGB',(left+len(selected)*w, top+len(stages)*h+16),PAPER)
    pen = ImageDraw.Draw(page)
    text(pen,15,12,'Five weak source images across distinct structures',32,True)
    text(pen,16,60,'Fixed 104 views · same global translation per build · green correct, blue missing, red extra',18,False,MUTED)
    for i,(_,label) in enumerate(stages):
        y = top+i*h
        text(pen,12,y+24,label,18,True)
        pen.line((8,y,page.width-10,y),fill='#c0cfcb',width=2)
    # Match the overlay to the recorded, visible pixels of each missed cube.
    priority_image = Image.new('RGB',(1260,485),PAPER)
    priority_pen = ImageDraw.Draw(priority_image)
    text(priority_pen,15,10,'Which visible blocks are still missing?',28,True)
    text(priority_pen,16,50,'Blue: required miss. Gold: optional horizontal sliver. All other pixels show the source image.',19,False,MUTED)
    featured = ['Shopfront','Portico','Colonnade 01']
    for n,key in enumerate(selected.values()):
        split,frame,prior,control=jobs_by_key[key]
        r=by_key[key]
        l=names[key]
        assert not any(frozen[key][k] for k in ('returns','texture','outlines'))
        assert not local_release()[0][key]['methods']
        assert all(r[k]==control['relative_geometry'][k] for k in ('correct','extras','missed'))
        orig=read(suite.RELEASE)['examples']
        assert frame['files']['rgb.png'] not in {ex['rgb'] for ex in orig}
        peers=[by_key[k] for k in l['frame_ids']]
        hits=sum(v['correct'] for v in peers)
        layout_f1=2*hits/(2*hits+sum(v['missed']+v['extras'] for v in peers))
        rep=dict(id=key,split=split,rgb_sha256=r['rgb_sha256'],pack_sha256=control['pack_sha256'],
                 f1=float(view_f1(control['relative_geometry'])))
        info=dict(name=l['name'],shape_sha256=l['shape_sha256'],rank=n+1,frames=len(peers),
                  f1=layout_f1,representative=rep)
        panels, evidence=sheet.prepare(suite.CONTROL,info,(frame,prior,control),atlas)
        traces.append(evidence)
        x=left+n*w
        text(pen,x,104,f"{n+1}. {l['name']}   {rep['f1']:.2%}",22,True)
        for i,(idx,_) in enumerate(stages):
            photo, inset, caption = panels[idx]
            caption=caption.replace('Photo / atlas','Source / atlas')
            if idx==0:
                caption=f"{r['correct']} correct, {r['missed']} missing, {r['extras']} extra. Layout average (pooled) {layout_f1:.2%}."
            sheet.panel(page,(x,top+i*h+8,w-13,h-14),(photo,inset,caption))
        image,_,_,_,_,labels,ids,truth,_,scored,_ = sheet.stages.read_view(suite.CONTROL,split,frame,prior,control)
        owners,_,_,bits,square,_ = block_footprints(labels,ids)
        lut=np.zeros(len(bits),np.uint8)
        for b in labels['blocks']:
            if tuple(b['xyz_world']) in truth-scored:
                i=b['id']
                lut[i]=2 if bits[i] in (1,2) and square[i]<4 else 1
        data=np.asarray(image).copy()
        classes=lut[owners[ids]]
        for i,color in [(1,(33,148,255)),(2,(255,188,40))]:
            data[classes==i]=.25*data[classes==i]+.75*np.asarray(color)
        bounds=Image.fromarray(ids>0).getbbox()
        prefix=f'{n+1:02}'
        for stage,idx in [('source',0),('outlines',1),('grid',5),('build',7),('errors',9)]:
            panels[idx][0].save(overlays/f'{prefix}-{stage}.jpg',quality=86,optimize=True)
        Image.fromarray(data).save(overlays/f'{prefix}-required.jpg',quality=89,optimize=True)
        if l['name'] in featured:
            focused=sheet.detail(Image.fromarray(data),bounds)
            x=featured.index(l['name'])*420+10
            text(priority_pen,x,101,l['name'],25,True)
            thumb=ImageOps.contain(focused,(400,260),Image.Resampling.LANCZOS)
            priority_image.paste(thumb,(x+(400-thumb.width)//2,145+(260-thumb.height)//2))
            text(priority_pen,x,420,f"{np.sum(lut==1)} required misses; {np.sum(lut==2)} optional misses",18,False)
            text(priority_pen,x,451,f"{r['extras']} extras in the export",17,False,MUTED)
    priority_image.save(OUT/'wall-examples.jpg',quality=94,subsampling=0)
    page.save(OUT/'stage-map.jpg',quality=91,subsampling=0,optimize=True)
    save(OUT/'stage-examples.json',dict(source_revision='abfc0be',ranking='lowest individual views, first per distinct layout',views=traces))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    jobs=list(suite.jobs(list(suite.framed.EVALUATION)))
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(grade_faces,jobs))
    assert len(rows)==104 and sum(r['missed'] for r in rows)==3083
    save(OUT/'face-opportunity.json',dict(source_revision='abfc0be',suite_cases_sha256=digest(
        ROOT/'facade-local-placement-01/cases.json'),frames=104,missed=3083,
        missed_by_visible_faces={k:sum(r['missed_by_visible_faces'][k] for r in rows) for k in ('1','2','3')},
        broad_and_smaller=sum(r['broad_and_smaller'] for r in rows),rows=rows))
    draw_examples(jobs,rows)


if __name__=='__main__':
    main()
