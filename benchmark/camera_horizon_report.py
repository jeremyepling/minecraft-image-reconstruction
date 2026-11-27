"""Verify unchanged exports, aggregate three independent floor-camera experiments and draw evidence."""
from collections import Counter
from itertools import product
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_horizon_run import BASE, CONTROL, OUT, REVIEW, COHORTS, SOURCE, frames, digest, read, save
from benchmark.camera_metrics import registration
from benchmark.expanded_scores import summarize
from engine.geometry import project
from benchmark.mask_blocks import decoded_pack
from benchmark.typed_structures import aligned_sites


ARMS={'free':'free','boundary':'boundary','source':'repeat','repeat_no_boundary':'grid'}


def overlay(entry, camera, case):
    image=Image.open(Path(entry['frame'])/'rgb.png').convert('RGB')
    draw=ImageDraw.Draw(image)
    P=np.asarray(camera['P'])
    pose=read(Path(entry['frame'])/'camera.json')
    T=registration(P,pose)
    labels=read(Path(entry['frame'])/'labels.json')
    aligned=P@np.linalg.inv(T)
    visible={tuple(c):xy for f in labels['faces'] for c,xy,ok in
             zip(f['corners_world'],f['corners_px'],f['corner_visible']) if ok}
    for corner,xy in list(visible.items())[::max(1,len(visible)//120)]:
        u,v=xy
        x,y=project(aligned,[corner])[0]
        draw.ellipse((u-2,v-2,u+2,v+2),outline='cyan',width=1)
        draw.line((x-2,y,x+2,y),fill='orange',width=1)
        draw.line((x,y-2,x,y+2),fill='orange',width=1)
    horizon=camera['evidence']['horizon_px']
    draw.line((0,horizon,image.width,horizon),fill='yellow',width=2)
    text=f"{entry['id']} | yellow: fitted horizon | cyan: visible truth corner | orange: projected corner"
    draw.rectangle((0,0,len(text)*7+12,26),fill='black')
    draw.text((6,6),text,fill='white')
    image.save(REVIEW/'example.jpg',quality=92)


def aggregate_validation(rows):
    if not rows:
        return dict(n=0)
    result=dict(n=len(rows))
    for name in ('fov_error_deg','ground_height_error_blocks','relative_scale_error',
                 'median_reprojection_px','p90_reprojection_px','axis_errors_deg'):
        vals=[max(row[name]) if name=='axis_errors_deg' else abs(row[name]) for row in rows if row[name] is not None]
        result[name]=dict(median=float(np.median(vals)),max=float(max(vals)))
    result['pooled_visible_corners_within_2px']=sum(r['within_2px'] for r in rows)
    result['pooled_visible_corners']=sum(r['vertices'] for r in rows)
    return result


def origin_sensitivity(entry, camera):
    folder=Path(entry['frame'])
    P=np.asarray(camera['P'])
    T=registration(P,read(folder/'camera.json'))
    vertices={tuple(c):xy for face in read(folder/'labels.json')['faces'] for c,xy,seen in
              zip(face['corners_world'],face['corners_px'],face['corner_visible']) if seen}
    if not vertices:
        return None
    attempts=[]
    for dx,dz in product((-1,0,1),repeat=2):
        adjusted=T.copy()
        adjusted[0,3]+=dx
        adjusted[2,3]+=dz
        xy=project(P@np.linalg.inv(adjusted),list(vertices))
        distances=np.linalg.norm(xy-list(vertices.values()),axis=1)
        attempts.append((float(np.percentile(distances,90)),dx,dz))
    error,dx,dz=min(attempts)
    return dict(evaluation_only=True,xz_shift=[dx,dz],p90_reprojection_px=error)


def compare_repeats(record, folder):
    candidate, other=record['arms']['source'], record['arms']['repeat_no_boundary']
    a, b=candidate['camera'], other['camera']
    if a['P'] is None or b['P'] is None:
        return None
    result={}
    names=[]
    for mode in ('repeat','grid'):
        case=read(folder/(mode+'-score.json'))
        pack=folder/mode/'reconstruction.zip'
        if not pack.exists() and mode=='grid' and a['P']==b['P']:
            pack=folder/'repeat'/'reconstruction.zip'
        if digest(pack)!=case['pack_sha256']:
            raise ValueError('repeat ablation pack differs from scored bytes')
        cells=decoded_pack(pack)
        mapped=aligned_sites(list(cells),np.array(case['registration']['transform']),case['translation'])
        names.append(dict(zip(map(tuple,mapped),cells.values())))
    left,right=names
    return dict(identical_matrix=a['P']==b['P'],horizon_abs_delta_px=abs(a['evidence']['horizon_px']-b['evidence']['horizon_px']),
                identical_aligned_typed_placement=left==right,
                source_only_positions=len(left.keys()-right.keys()),grid_only_positions=len(right.keys()-left.keys()),
                shared_positions_changed_id=sum(left[p]!=right[p] for p in left.keys()&right.keys()))


def main():
    controls=CONTROL/'benchmark/reviews/expanded-baseline-01'
    base=read(controls/'cases.json')
    if digest(controls/'cases.json')!=read(controls/'summary.json')['cases_sha256']:
        raise ValueError('frozen baseline cases changed')
    entries={e['id']:e for e in frames(SOURCE,COHORTS)}
    groups={arm:[] for arm in ['baseline',*ARMS]}
    diagnostics=[]
    fixed=CONTROL/'benchmark/reviews/typed-structures-01/typed-exports.zip'
    if digest(fixed)!=read(controls/'manifest.json')['fixed']['typed_exports_sha256']:
        raise ValueError('fixed export release changed')
    REVIEW.mkdir(parents=True,exist_ok=True)
    example=None
    with ZipFile(fixed) as archive:
        for row in base:
            id=row['id']
            groups['baseline'].append(row)
            if row['dataset']=='fixed-104':
                pack_hash=sha256(archive.read(f'{id}/reconstruction.zip')).hexdigest()
            else:
                pack_hash=digest(BASE/row['dataset']/id/'reconstruction.zip')
            if pack_hash!=row['pack_sha256']:
                raise ValueError(f'Baseline pack mismatch for {id}')
            record={k:row[k] for k in ('id','dataset','split','source_sha256','pack_sha256','camera_reason','atlas_camera_reason')}
            record['ordinary_supported']=row['camera_usable']
            record['arms']={}
            for arm,mode in ARMS.items():
                scored=row
                if not row['camera_usable']:
                    folder=OUT/id
                    cached=read(folder/(mode+'.json'))
                    if cached['rgb_sha256']!=row['source_sha256']:
                        raise ValueError('prediction used different RGB')
                    camera=cached['camera']
                    if camera['P'] is not None:
                        scored=read(folder/(mode+'-score.json'))
                        if scored['source_sha256']!=row['source_sha256']:
                            raise ValueError('candidate result used different RGB')
                        if arm=='source' and row['split']!='train' and (example is None or scored['exported']>example[2]['exported']):
                            example=(entries[id],camera,scored)
                    if arm=='source':
                        grid=read(folder/'grid.json')
                        if grid['rgb_sha256']!=row['source_sha256']:
                            raise ValueError('grid proposal used different RGB')
                        record['repeat_grid_only_camera']=grid['camera']
                        if camera['P'] is not None:
                            record['origin_sensitivity']=origin_sensitivity(entries[id],camera)
                    record['arms'][arm]=dict(camera=camera,pack_sha256=scored['pack_sha256'],
                        geometry=scored['geometry'],scores=scored['scores'],camera_validation=scored.get('camera_validation'))
                groups[arm].append(scored)
            if not row['camera_usable']:
                record['repeat_export_comparison']=compare_repeats(record,OUT/id)
            diagnostics.append(record)
    summaries={arm:summarize(rows,list(entries.values())) for arm,rows in groups.items()}
    datasets={}
    for dataset in ('fixed-104',*COHORTS.keys()):
        datasets[dataset]={}
        for arm in groups:
            result=summaries[arm][dataset]
            rows=[r for r in groups[arm] if r['dataset']==dataset and 'camera_validation' in r]
            result['new_camera_validation']=aggregate_validation([r['camera_validation'] for r in rows])
            for split,sr in result['splits'].items():
                sr['new_camera_validation']=aggregate_validation([r['camera_validation'] for r in rows if r['split']==split])
            datasets[dataset][arm]=result
    save(REVIEW/'cases.json.gz',diagnostics)
    save(REVIEW/'summary.json',dict(datasets=datasets,base_revision='7f63d4a',
         control_cases_sha256=digest(controls/'cases.json'),cases_sha256=digest(REVIEW/'cases.json.gz'),
         gates=dict(total=len(base),ordinary_supported=sum(r['camera_usable'] for r in base)),
         repeat_export_comparisons=[dict(id=r['id'],**r['repeat_export_comparison']) for r in diagnostics if r.get('repeat_export_comparison')],
         repeat_horizon_ablation={dataset:dict(Counter(
             ('both' if r['arms']['source']['camera']['P'] is not None and r['repeat_grid_only_camera']['P'] is not None else
              'boundary_grid_only' if r['arms']['source']['camera']['P'] is not None else
              'free_grid_only' if r['repeat_grid_only_camera']['P'] is not None else 'neither')
             for r in diagnostics if not r['ordinary_supported'] and r['dataset']==dataset)) for dataset in COHORTS},
         fallback_attempts=118, source_arm='repeat-pattern translations with weak boundary and grid horizon proposals'))
    if example:
        overlay(*example)
    lines=['# Horizon and floor-repeat camera experiment','',
        'The ordinary camera gets the first decision on each source image. These three fallback arms apply only on its 118 abstentions. '
        'The free and boundary arms use scalar-frequency shortlists and independently checked 2-D phase gradients. '
        'The free arm sweeps possible horizons. The boundary arm limits the search to horizons near a long transition at both image borders.',
        '',
        'The source arm measures the repeated 16-texel motif directly. It rectifies the lower image at trial horizons, finds two independent '
        'brightness translations, and refines both across separated patches. A border transition adds hypotheses to a free grid. '
        'Every acceptance requires the ordinary same-sprite, orientation and origin check across both halves, and a metric camera lift. '
        'Each newly supported camera was sent through the ordinary whole build and its exported ZIP was decoded. Recovered cameras were scored against '
        'the recorded pose and visible corners only after prediction. The test uses vanilla Java 26.3 concrete staging floors.',
        '',
        '| Cohort | Arm | Camera supported | Strict geometry F1 | Facade geometry F1 | Strict position + ID F1 | Facade position + ID F1 | Correct / missed / extra positions |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset,arms in datasets.items():
        for arm,r in arms.items():
            g=r['geometry']
            s=r['scores']
            t=g['strict']
            lines.append(f"| {dataset} | {arm} | {r['camera']['usable']}/{r['tested']} | {t['f1']:.4f} | "
                         f"{g['facade']['f1']:.4f} | {s['strict']['f1']:.4f} | {s['facade']['f1']:.4f} | {t['correct']} / {t['missed']} / {t['extras']} |")
    lines.extend(['','All original source-supported build exports were carried forward and verified by SHA-256. '
                  'The fixed 104 received no fallback attempts. Every abstention stays in the overall denominator. '
                  'See summary.json for frozen splits, matched-position material accuracy, and conditional camera errors. '
                  'cases.json.gz contains every camera and decision. The example overlays estimated and recorded visible corners on the original input.',
                  '', 'Run these commands from this checkout with the frozen cohorts in the stable runs directory and the full-expanded-controls worktree beside it.',
                  '', '```powershell','python -m benchmark.camera_horizon_run predict','python -m benchmark.camera_horizon_run build',
                  'python -m benchmark.camera_horizon_run repeat-predict','python -m benchmark.camera_horizon_run repeat-build',
                  'python -m benchmark.camera_horizon_run grid-predict','python -m benchmark.camera_horizon_run grid-build','python -m benchmark.camera_horizon_report','```'])
    (REVIEW/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    main()


