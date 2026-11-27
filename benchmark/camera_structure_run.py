"""Run the failure-only scaffold challenger and compare unchanged ordinary full exports."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import argparse
import sys

from benchmark.camera_structure_scaffold import infer, line_camera
from benchmark.expanded_baseline import ROOT,SOURCE,COHORTS,digest,frames,read,save
from benchmark.expanded_scores import summarize
from benchmark.camera_metrics import truth_camera
from engine.surface_model import intrinsic
import numpy as np
from PIL import Image,ImageDraw

BASE=ROOT.parent/'full-expanded-controls/runs/expanded-baseline-01'
RUN=ROOT/'runs/camera-structure-scaffold-02'
REPORT=ROOT/'benchmark/reviews/camera-structure-scaffold-01'


def orientation(linear, pose):
    if linear is None:
        return None
    size=pose['image_size']
    truth=truth_camera(pose)[:,:3]
    def basis(matrix):
        matrix=matrix/np.linalg.norm(matrix[2])
        focal=np.linalg.norm(matrix[0]-.5*size[0]*matrix[2])
        return np.linalg.solve(intrinsic(size,focal),matrix),focal
    a,f=basis(np.asarray(linear))
    b,g=basis(truth)
    cosine=np.abs(b.T@a)
    options=[cosine[[0,1,2],[0,1,2]],cosine[[2,1,0],[0,1,2]]]
    errors=np.degrees(np.arccos(np.clip(max(options,key=sum),0,1)))
    return dict(max_axis_deg=float(errors.max()),axis_deg=errors.tolist(),focal_ratio=float(f/g))


def paths(e):
    return BASE/e['dataset']/e['id'],RUN/e['dataset']/e['id']


def predict(entry):
    base,out=paths(entry)
    img=Path(entry['frame'])/'rgb.png'
    receipt=read(base/'receipt.json')
    assert digest(img)==entry['rgb_sha256']==receipt['source_sha256']
    assert digest(base/'reconstruction.zip')==receipt['pack_sha256']
    image=Image.open(img).convert('RGB')
    lines=read(base/'handoffs/mlsd.json.gz')['segments']
    source=read(base/'handoffs/placement-input.json.gz')
    camera,info=infer(image,source['parts'],lines,source['counts'])
    save(out/'source.json.gz',dict(camera=camera,**info))
    # Evaluation begins after freezing the source decision.
    orig=line_camera(lines,image.size)
    pose=read(Path(entry['frame'])/'camera.json')
    grade=dict(ordinary_lines=orientation(orig,pose),structure=orientation(info['linear'],pose),
               scaffold=bool(info.get('scaffold')),reason=info['reason'],accepted=camera is not None)
    grade['gates']=info.get('gates',{})
    grade['tentative']=[]
    for option in info.get('tentative',[]):
        from calibration.world import GROUND
        P=np.asarray(option['P'])
        height=-np.linalg.solve(P[:,:3],P[:,3])[1]
        grade['tentative'].append(dict(kind=option['kind'],height_ratio=float(height/(pose['eye_world'][1]-GROUND-1))))
    save(out/'angles.json',grade)
    return entry['id'],grade['reason']


def replay(entry):
    base,out=paths(entry)
    camera=read(out/'source.json.gz')['camera']
    if camera:
        # Reuse the exact metric stage injection shared with the floor challenger.
        sys.path.append(str(ROOT.parent/'camera-horizon-floor/benchmark'))
        from camera_horizon_replay import replay as full_build
        result=full_build(entry,camera,base,out/'build')
        save(out/'graded.json',result)
        return entry['id'],result['exported']
    return entry['id'],0


def distribution(rows):
    summary={}
    for arm in ('ordinary_lines','structure'):
        measurements=[r[arm] for r in rows if r[arm]]
        def median(vals):
            return float(np.median(vals)) if vals else None
        summary[arm]=dict(returned=len(measurements),axis_median_deg=median([x['max_axis_deg'] for x in measurements]),
            focal_ratio_median=median([x['focal_ratio'] for x in measurements]),
            within_2deg_and_5pct=sum(x['max_axis_deg']<=2 and abs(x['focal_ratio']-1)<=.05 for x in measurements))
    paired=[r for r in rows if r['ordinary_lines'] and r['structure']]
    return dict(tested=len(rows),**summary,paired=len(paired),
                axis_improved=sum(r['structure']['max_axis_deg']<r['ordinary_lines']['max_axis_deg'] for r in paired),
                gates=dict(sum((Counter(r.get('gates',{})) for r in rows),Counter())),
                tentative_heights=[m for r in rows for m in r.get('tentative',[])],
                reasons=dict(Counter(r['reason'] for r in rows)))


def aggregate(entries,controls):
    ids={(e['dataset'],e['id']):e for e in entries}
    rows=[]
    chosen=[]
    angle={}
    for base in controls:
        k=base['dataset'],base['id']
        alt=base
        ev=None
        if k in ids and not base['camera_usable']:
            e=ids[k]
            _,out=paths(e)
            ev=read(out/'angles.json')
            if ev['accepted']:
                alt=read(out/'graded.json')
            angle[k]=ev
        rows.append(dict(dataset=k[0],id=k[1],split=base['split'],baseline=base,source=alt,evidence=ev))
        chosen.append(alt)
    summary={}
    for arm,data in [('baseline',controls),('source',chosen)]:
        for ds,item in summarize(data,entries).items():
            summary.setdefault(ds,{})[arm]=item
    for ds,item in summary.items():
        group=[r for r in rows if r['dataset']==ds and r['evidence']]
        item['angles']=distribution([r['evidence'] for r in group])
        item['angles_by_split']={s:distribution([r['evidence'] for r in group if r['split']==s]) for s in sorted({r['split'] for r in group})}
        item['angles_by_layout']={layout:distribution([r['evidence'] for r in group if r['baseline']['layout']==layout]) for layout in sorted({r['baseline']['layout'] for r in group})}
        item['delta']=dict(recovered=sum(r['source']['geometry']['strict']['correct'] for r in group),
                           lost=0,extra_added=sum(r['source']['geometry']['strict']['extras'] for r in group),extra_removed=0)
    save(REPORT/'summary.json',summary)
    save(REPORT/'cases.json.gz',rows)
    save(REPORT/'source-fits.json.gz',[dict(dataset=e['dataset'],id=e['id'],**read(paths(e)[1]/'source.json.gz')) for e in entries if (e['dataset'],e['id']) in angle])
    active=[e for e in entries if (e['dataset'],e['id']) in angle and angle[e['dataset'],e['id']]['scaffold']]
    if active:
        e=max(active,key=lambda e:(angle[e['dataset'],e['id']]['accepted'],len(read(paths(e)[1]/'source.json.gz')['scaffold']['edges'])))
        source=Image.open(Path(e['frame'])/'rgb.png').convert('RGB')
        draw=ImageDraw.Draw(source)
        info=read(paths(e)[1]/'source.json.gz')
        scaffold=info['scaffold']
        for a,b,axis in scaffold['edges']:
            draw.line([tuple(scaffold['xy'][a]),tuple(scaffold['xy'][b])],fill=['#ffad43','#3debb1','#ff51c3'][axis],width=4)
        for a in info.get('anchors',[]):
            q=[tuple(p) for p in a['quad']]
            draw.line(q+[q[0]],fill='white',width=2)
        source.thumbnail((1080,675))
        source.save(REPORT/'scaffold.jpg',quality=88)
        save(REPORT/'example.json',{k:e[k] for k in ('id','split','dataset','rgb_sha256')})
    save(REPORT/'manifest.json',dict(revision='7f63d4a',baseline_cases=digest(BASE/'scores/cases.json'),
         inputs=[{k:e[k] for k in ('dataset','id','rgb_sha256','split','dataset_manifest_sha256')} for e in entries],
         replay='camera_horizon_replay.replay',replay_sha256=digest(ROOT.parent/'camera-horizon-floor/benchmark/camera_horizon_replay.py'),predictor_sha256=digest(Path(__file__).with_name('camera_structure_scaffold.py'))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['predict','replay','aggregate'])
    parser.add_argument('--workers',type=int,default=3)
    opt=parser.parse_args()
    controls=read(BASE/'scores/cases.json')
    entries=list(frames(SOURCE,COHORTS))
    if opt.task=='aggregate':
        return aggregate(entries,controls)
    failed={(r['dataset'],r['id']) for r in controls if not r['camera_usable']}
    todo=[e for e in entries if (e['dataset'],e['id']) in failed]
    if opt.task=='predict':
        todo=[e for e in todo if not (paths(e)[1]/'angles.json').exists()]
    else:
        todo=[e for e in todo if read(paths(e)[1]/'angles.json')['accepted'] and not (paths(e)[1]/'graded.json').exists()]
    with ProcessPoolExecutor(opt.workers) as pool:
        for item in pool.map(predict if opt.task=='predict' else replay,todo):
            print(item,flush=True)


if __name__=='__main__':
    main()
