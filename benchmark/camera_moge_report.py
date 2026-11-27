"""Summarize identical source images and render scored native visible-corner examples."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from benchmark.camera_moge_fields import BASE,RUN,ROOT,WORKTREES,failures,read,save,digest
from benchmark.camera_metrics import registration
from benchmark.expanded_scores import summarize
from engine.geometry import project

REPORT=ROOT/'benchmark/reviews/camera-moge-surface-first-01'
HORIZON=WORKTREES/'camera-horizon-floor/runs/camera-horizon-floor-01'


def overlay(entry,proposal,case):
    src=Path(entry['frame'])
    image=Image.open(src/'rgb.png').convert('RGB')
    marks=ImageDraw.Draw(image)
    for a in proposal['anchors']:
        poly=[tuple(v) for v in a['quad']]
        marks.line(poly+[poly[0]],fill='#ffe359',width=3)
    pose=read(src/'camera.json')
    P=np.asarray(proposal['camera']['P'])
    aligned=P@np.linalg.inv(registration(P,pose))
    corners={tuple(c):xy for f in read(src/'labels.json')['faces']
             for c,xy,v in zip(f['corners_world'],f['corners_px'],f['corner_visible']) if v}
    keys=sorted(corners)
    for c in keys[::max(1,len(keys)//22)]:
        xy=corners[c]
        xp=project(aligned,np.array([c]))[0]
        marks.line([tuple(xy),tuple(xp)],fill='#f77b48',width=2)
        for point,color in [(xy,'#59efbf'),(xp,'#f77b48')]:
            x,y=point
            marks.ellipse((x-4,y-4,x+4,y+4),fill=color)
    image.thumbnail((900,600),Image.Resampling.LANCZOS)
    panel=Image.new('RGB',(image.width,image.height+50),'#162828')
    panel.paste(image,(0,50))
    d=ImageDraw.Draw(panel)
    d.text((10,4),entry['id']+'   |   Yellow: source atlas matches',fill='white',font_size=16)
    d.text((10,25),'Green: native visible corner   |   Orange: projected corner   |   Median error %.1f px'%case['camera_validation']['median_reprojection_px'],fill='white',font_size=15)
    panel.save(REPORT/(entry['id']+'.jpg'),quality=92)


def main():
    REPORT.mkdir(exist_ok=True,parents=True)
    entries=failures()
    controls=read(BASE/'scores/cases.json')
    index={(e['dataset'],e['id']):e for e in entries}
    frozen_path=WORKTREES/'camera-model-controls/benchmark/reviews/camera-model-controls-01/frames.json'
    frozen={(r['dataset'],r['id']):r for r in read(frozen_path)}
    if len(frozen)!=len(controls) or len(controls)!=288:
        raise ValueError('Control cohort mismatch')
    rows=[]
    arms=defaultdict(list)
    for b in controls:
        key=b['dataset'],b['id']
        repeat_path=HORIZON/b['id']/'repeat-score.json'
        repeat=read(repeat_path) if repeat_path.exists() else b
        frame=frozen[key]
        if (frame['rgb_sha256']!=b['source_sha256'] or frame['split']!=b['split'] or
            frame['ordinary_supported']!=b['camera_usable'] or frame['repeat_supported']!=repeat['camera_usable']):
            raise ValueError(f'Frozen control mismatch: {key}')
        model=b
        if key in index:
            out=RUN/key[0]/key[1]
            p=read(out/'proposal.json')
            if p['camera']:
                model=read(out/'case.json')
                if digest(out/'build/reconstruction.zip')!=model['pack_sha256']:
                    raise ValueError(f'Export digest mismatch: {key}')
                if b['id'] in ('test-castle-03-limestone-rear-oblique','train-amphitheater-02-ashlar-rear-oblique'):
                    overlay(index[key],p,model)
            rows.append(dict(dataset=b['dataset'],id=b['id'],split=b['split'],source_sha256=b['source_sha256'],
                reason=p['reason'],orientation=read(out/'angle.json'),scale_measurements=p.get('scales',[]),
                linear=p['linear'],projection=p['camera'], model_field_sha256=digest(out/'field.npz'),
                anchors=len(p.get('anchors',[])),repeat_supported=bool(repeat['camera_usable']),
                result=model if p['camera'] else None))
        for name,r in [('ordinary',b),('repeat',repeat),('moge',model),('repeat_then_moge',repeat if repeat['camera_usable'] else model)]:
            arms[name].append(r)
    summary={k:summarize(v,controls) for k,v in arms.items()}
    save(REPORT/'scores.json',summary)
    save(REPORT/'proposals.json.gz',rows)
    from benchmark.moge_probe import SHA256,REVISION
    save(REPORT/'manifest.json',dict(base='d9eb377',branch='experiment/camera-moge-surface-first',
        frames=288,control_frames_sha256=digest(frozen_path),control_cases_sha256=digest(BASE/'scores/cases.json'),
        checkpoint_sha256=SHA256,checkpoint_revision=REVISION,
        atlas_sha256='49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950',
        code_sha256={p.name:digest(p) for p in (ROOT/'benchmark').glob('camera_moge_*.py')},
        scores_sha256=digest(REPORT/'scores.json'),proposals_sha256=digest(REPORT/'proposals.json.gz')))
    print('method,dataset,strict,facade,ID strict,ID facade,cameras rejected,TP,FN,FP')
    for name,ds in summary.items():
        for cohort,s in ds.items():
            geo,typed=s['geometry'],s['scores']
            print(name,cohort,*[round(100*t[v]['f1'],2) for t in (geo,typed) for v in ('strict','facade')],s['camera']['unusable'],*[geo['strict'][k] for k in ('correct','missed','extras')])
    print('per-case metrics for 5 incremental frames')
    for r in rows:
        if r['result'] and not r['repeat_supported']:
            c=r['result']
            print(r['id'],c['geometry']['strict'],c['camera_validation']['median_reprojection_px'])
    angles=[r['orientation'] for r in rows if r['orientation']]
    print('orientation',len(angles),'worst axis median deg',np.median([a['max_axis_deg'] for a in angles]),'pass tolerance', sum(a['max_axis_deg']<=2 and abs(a['focal_ratio']-1)<=.05 for a in angles))
    print('reasons',Counter(r['reason'] for r in rows))

if __name__=='__main__':
    main()
