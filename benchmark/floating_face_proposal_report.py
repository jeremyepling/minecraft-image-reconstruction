"""Score frozen offers against projected visible cube faces and decode paired placements."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from benchmark.camera_labels import screen_matrix
from benchmark.floating_face_proposal_run import OUT, GROUPS, ROOT, source_inputs
from benchmark.floating_patch_controls import read, save, digest
from benchmark.floating_patch_report import baseline
from benchmark.floating_bridges_report import summarize
from benchmark.full_block_lattice import atlas
from benchmark.relative_face_assembly import oriented, axis_vector
from benchmark.relative_face_scoring import _yaw
from engine.geometry import UNIT, project

REVIEW = ROOT/'benchmark/reviews/floating-face-proposals-01'
EXTRA = ROOT.parent/'floating-face-audit/runs/floating-holdout3'


def extra_root(group):
    return EXTRA.with_name('floating-holdout4') if group=='new7' else EXTRA


def reference(frame):
    labels = read(frame/'labels.json')
    cam = read(frame/'camera.json')
    matrix = screen_matrix(cam)
    eye = np.asarray(cam['eye_world'])
    with np.load(frame/'labels.npz') as array:
        owners = array['face_id']
    counts = np.bincount(owners.ravel().astype(int))
    blocks = {b['id']:b for b in labels['blocks']}
    faces = []
    for face in labels['faces']:
        block = blocks[face['block']]
        if face['normal_world'][1] or block.get('geometry_kind','full_cube') != 'full_cube':
            continue
        quad = np.asarray(face['corners_px'],'f')
        camera = (np.asarray(face['corners_world'])-eye) @ matrix[:,:3].T+matrix[:,3]
        assert np.max(np.abs(camera[:,:2]/camera[:,2:]-quad)) < .002
        assert counts[face['id']] == face['visible_pixels']
        area = cv2.contourArea(quad)
        if area<16 or counts[face['id']]<.90*area:
            continue
        if (quad<0).any() or (quad>owners.shape[::-1]).any():
            continue
        faces.append((face['id'],area,quad,block['block_id']))
    if not faces:
        return faces,np.empty((0,4))
    rects = np.array([np.r_[q.min(0),q.max(0)] for _,_,q,_ in faces])
    return faces,rects


def match(quad, truth):
    poly = np.array(quad,'f')
    area = cv2.contourArea(poly)
    low,high = poly.min(0),poly.max(0)
    faces,rects = truth
    nearby = (rects[:,:2]<high).all(1) & (rects[:,2:]>low).all(1)
    best = dict(iou=0.,face=None,material=None)
    for j in np.flatnonzero(nearby):
        fid,refarea,ref,block = faces[j]
        common,_ = cv2.intersectConvexConvex(poly,ref)
        iou = float(common/max(.01,area+refarea-common))
        if iou > best['iou']:
            best = dict(iou=iou,face=fid,material=block,quad=ref.tolist())
    return best


def transferred_yaw(before, after, yaw):
    originals={c['id']:c for c in oriented(before)[0]}
    turns=[]
    for c in oriented(after)[0]:
        old=originals.get(c['id'])
        if old is None or old['normal_camera']!=c['normal_camera']:
            continue
        # A shared measured wall normal has one signed horizontal bearing under a four-way yaw.
        match=[i for i in range(4) if np.array_equal(_yaw(axis_vector(c['normal_axis']),i),axis_vector(old['normal_axis']))]
        assert len(match)==1
        turns.append(match[0])
    assert len(set(turns))==1
    return (yaw+turns[0])%4


def eval_faces(job):
    group,c = job
    _,raw,whole,_=source_inputs(group,extra_root(group))
    name=c['id']
    truth=reference(Path(c['frame']))
    pack=read(OUT/group/'offers'/(name+'.json.gz'))
    assert pack['source_sha256']==digest(Path(c['variants']['sky']['image']))
    rel=Path('sky')/name/'charts.json'
    seed={v['id']:v for v in read(raw/rel)['charts']}
    packet=read(whole/rel)
    charts=[]
    for row in pack['charts']:
        chart=seed[row['id']]
        seeds=[dict(quad=project(np.array(chart['H']),UNIT+[cell['u'],cell['v']]).tolist(),
                    atlas=cell['block_id'],score=cell.get('confidence',0)) for cell in chart['cells']]
        def attach(offers):
            return [dict(o,truth=match(o['quad'],truth)) for o in offers]
        olds=attach(seeds)
        offers=attach(row['offers'])
        best=lambda a:max(a,key=lambda o:o['truth']['iou']) if a else None
        source_seed=max(olds,key=lambda o:o['score'])
        pick=None if row['picked'] is None else offers[row['picked']]
        charts.append(dict(id=row['id'],offers=len(offers),accepted=bool(pick),source_seed=source_seed,
                           best_seed=best(olds),best_offer=best(offers),selected_offer=pick,
                           effective=pick or source_seed,before=row['before'],after=row['after']))
    scorepath=Path('connected-atlas/sky')/name/'score.json'
    oldroot=extra_root(group)/'horizontal-cells/assembly' if group in ('new6','new7') else baseline(group,True)
    manifests=[read(root/scorepath) for root in (oldroot,OUT/group/'assembly')]
    assert manifests[0]['reference_sha256']==manifests[1]['reference_sha256']
    for root,doc in zip((whole,OUT/group/'charts'),manifests):
        assert doc['source_sha256']==pack['source_sha256']
        if doc['score']['fixed'] is None:
            ordinary=read(source_inputs(group,extra_root(group))[0]/'assembly'/scorepath)['score']['fixed']
            compass=transferred_yaw(list(seed.values()),read(root/rel)['charts'],ordinary['yaw_quarters'])
            assert compass==doc['score']['relaxed_yaw']['yaw_quarters']
    provenance={k:packet[k] for k in ('source_sha256','surfaces_sha256','counts_sha256')}
    provenance.update(ordinary_chart=digest(raw/rel),whole_chart=digest(whole/rel),
                      measured_chart=manifests[1]['chart_sha256'],packs=[d['pack_sha256'] for d in manifests],
                      fixed_yaw=[d['score']['fixed'] is not None for d in manifests],reference_sha256=manifests[1]['reference_sha256'])
    return dict(id=name,group=group,charts=charts,provenance=provenance)


def examples(rows):
    examples=[r for r in rows if (r['group']=='sky' and 'chapel' in r['id']) or ('observatory' in r['id'] and r['group']=='supplemental')]
    if not examples:
        return
    sheet=Image.new('RGB',(840,250*len(examples)), '#152333')
    pen=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('arial.ttf',15)
    for idx,row in enumerate(examples):
        case=next(c for c in source_inputs(row['group'],EXTRA)[3] if c['id']==row['id'])
        image=Image.open(case['variants']['sky']['image']).convert('RGB')
        if not row['charts']:
            continue
        chart=max(row['charts'],key=lambda c:c['before'])
        full=image.copy()
        y=idx*250
        for col,(field,text) in enumerate((('source_seed','Old seed'),('effective','Source choice'),('best_offer','Best offered using truth'))):
            item=chart[field]
            if item is None:
                continue
            q=np.array(item['quad'])
            ref=item['truth'].get('quad')
            pts=q if ref is None else np.vstack([q,ref])
            center=pts.mean(0)
            side=max(24,.8*max(np.ptp(pts,axis=0)))
            bounds=(center[0]-side,center[1]-side,center[0]+side,center[1]+side)
            crop=image.crop(tuple(map(int,bounds))).resize((174,174),Image.Resampling.NEAREST)
            draw=ImageDraw.Draw(crop)
            for poly,color in ((q,'#ffd764'),(ref,'#48caff')):
                if poly is not None:
                    coords=[tuple(pt) for pt in (np.array(poly)-bounds[:2])*174/(2*side)]
                    draw.line(coords+coords[:1],fill=color,width=2)
            x=300+col*179
            sheet.paste(crop,(x,y+30))
            pen.text((x,y+7),text,font=font,fill='white')
            pen.text((x,y+210),f"IoU {item['truth']['iou']:.2f}",font=font,fill='white')
            if field=='effective':
                ImageDraw.Draw(full).rectangle(bounds,outline='#ffd764',width=5)
        full.thumbnail((290,175))
        sheet.paste(full,(3,y+28))
        pen.text((4,y+6),case['id'].split('-',1)[1],font=font,fill='white')
        pen.text((4,y+212),'Gold proposal   Blue visible cube face',font=font,fill='white')
    sheet.save(REVIEW/'examples.jpg',quality=92)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=6)
    cfg=cli.parse_args()
    groups=(*GROUPS,'new6','new7') if (OUT/'new7').exists() else (*GROUPS,'new6')
    jobs=[(g,c) for g in groups for c in source_inputs(g,extra_root(g))[3]]
    REVIEW.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(cfg.workers) as pool:
        rows=list(pool.map(eval_faces,jobs))
    save(REVIEW/'face-matches.json.gz',rows)
    save(REVIEW/'manifest.json',dict(frozen_source_revision='a4ce17a',base_revision='baff1db',
        atlas=atlas().provenance,method_files={name:digest(ROOT/'benchmark'/name) for name in
        ('floating_face_proposals.py','floating_face_proposal_run.py')},panels={g:[dict(id=c['id'],
        split=c['split'],dataset=c['dataset']) for group,c in jobs if group==g] for g in groups}))
    examples(rows)
    report={}
    for g in groups:
        prev=extra_root(g)/'horizontal-cells/assembly' if g in ('new6','new7') else baseline(g,True)
        report[g]=summarize(source_inputs(g,extra_root(g))[3],
                     {'horizontal':prev,'proposed':OUT/g/'assembly'},'connected-atlas/sky')
    save(REVIEW/'placement.json.gz',report)
    lines=['| Panel | Method | Strict | Facade | Strict + ID | Facade + ID | Correct / missed / extra | Gained / lost / extras added / removed |',
           '|---|---|---:|---:|---:|---:|---|---|']
    for g,v in report.items():
        for method,s in v['cohorts']['all'].items():
            metrics=[f"{100*s[k]['f1']:.2f}" for k in ('geometry_strict','geometry_facade','scores_strict','scores_facade')]
            counts=' / '.join(str(s['geometry_strict'][k]) for k in ('correct','missed','extras'))
            diffs=' / '.join(str(s['paired'][k]) for k in ('gained','lost','added','removed'))
            lines.append('| '+' | '.join([g,method,*metrics,counts,diffs])+' |')
    (REVIEW/'scores.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
