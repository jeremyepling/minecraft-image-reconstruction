"""Verify unchanged exports, aggregate floor-camera experiments and draw evidence."""
from collections import Counter
from itertools import product
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.camera_horizon_run import BASE, CONTROL, OUT, REVIEW, COHORTS, SOURCE, frames, digest, read, save
from benchmark.camera_metrics import registration, truth_camera
from benchmark.expanded_scores import summarize
from engine.geometry import project
from engine.camera_grid import plane_coordinates
from calibration.world import GROUND
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
    font=ImageFont.load_default(size=20)
    draw.rectangle((0,0,draw.textlength(text,font=font)+20,34),fill='black')
    draw.text((10,4),text,fill='white',font=font)
    # Paint the measured visible labels from the same frame after decoding and
    # rigidly registering this view's actual exported blocks.
    ground_truth = {tuple(block['xyz_world']) for block in labels['blocks']}
    local = decoded_pack(OUT / entry['id'] / 'repeat' / 'reconstruction.zip')
    world = set(aligned_sites(list(local), np.array(case['registration']['transform']), case['translation']))
    folder = Path(entry['frame'])
    raw = np.asarray(Image.open(folder/'rgb.png').convert('RGB')).copy()
    with np.load(folder/'labels.npz') as maps:
        face_ids = maps['face_id']
    colors = np.zeros((int(face_ids.max())+1, 3), dtype=np.uint8)
    for face in labels['faces']:
        if face['id'] < len(colors):
            colors[face['id']] = (29,189,110) if tuple(face['xyz_world']) in world else (235,65,96)
    visible = face_ids>0
    raw[visible] = .55*raw[visible]+.45*colors[face_ids[visible]]
    placement = Image.fromarray(raw)
    annotation = ImageDraw.Draw(placement)
    extras = list(world-ground_truth)
    if extras:
        xy = project(truth_camera(pose),np.array(extras)+.5)
        for x,y in xy:
            if 0<=x<image.width and 0<=y<image.height:
                annotation.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=(244,174,58))
    totals=case['geometry']['strict']
    caption=(f"Export: green {totals['correct']} correct | red {totals['missed']} missed | "
             f"amber dots {totals['extras']} extra (projected centers)")
    annotation.rectangle((0,0,annotation.textlength(caption,font=font)+20,34),fill='black')
    annotation.text((10,4),caption,fill='white',font=font)
    combined=Image.new('RGB',(image.width,2*image.height))
    combined.paste(image,(0,0))
    combined.paste(placement,(0,image.height))
    combined.save(REVIEW/'example.jpg',quality=91)


def aggregate_validation(rows):
    if not rows:
        return dict(n=0)
    result=dict(n=len(rows))
    for name in ('fov_error_deg','ground_height_error_blocks','relative_scale_error',
                 'median_reprojection_px','p90_reprojection_px','axis_errors_deg','ground_phase_median_blocks','ground_metric_span_relative_error'):
        vals=[max(row[name]) if name=='axis_errors_deg' else abs(row[name]) for row in rows if row[name] is not None]
        result[name]=dict(median=float(np.median(vals)),max=float(max(vals)))
    result['pooled_visible_corners_within_2px']=sum(r['within_2px'] for r in rows)
    result['pooled_visible_corners']=sum(r['vertices'] for r in rows)
    return result


def ground_validation(entry, camera):
    """Independently test native ground coordinate phase and one-tile metric scale."""
    pose=read(Path(entry['frame'])/'camera.json')
    w,h=pose['image_size']
    P=np.asarray(camera['P'])
    T=registration(P,pose)
    world=P@np.linalg.inv(T)
    actual=truth_camera(pose)
    xs,ys=np.meshgrid(w*np.array([.1,.3,.5,.7,.9]),h*np.array([.82,.87,.92,.97]))
    xy=np.column_stack((xs.ravel(),ys.ravel()))
    wanted=plane_coordinates(actual,xy,1,GROUND+1)
    got=plane_coordinates(world,xy,1,GROUND+1)
    delta=got-wanted
    error=np.linalg.norm(delta-np.rint(delta),axis=1)
    wanted_dist=np.linalg.norm(wanted[:,None]-wanted[None,:],axis=-1)
    got_dist=np.linalg.norm(got[:,None]-got[None,:],axis=-1)
    use=np.triu(wanted_dist>1.,k=1)
    return dict(ground_phase_median_blocks=float(np.median(error)),
                ground_metric_span_relative_error=float(np.median(abs(got_dist[use]/wanted_dist[use]-1))))


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
    reference_study={r['id']:r for r in read(controls/'camera-diagnosis.json')['cases']}
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
            record['baseline_geometry']=row['geometry']
            record['baseline_scores']=row['scores']
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
                        scored['camera_validation'].update(ground_validation(entries[id],camera))
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
                prior=reference_study[id]
                if prior['source_sha256']!=row['source_sha256']:
                    raise ValueError('reference floor study used a different source')
                record['reference_floor_sampler_accepted']=prior['true_atlas_accepted']
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
         code_sha256={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('camera_horizon*.py'))},
         input_dataset_sha256=read(controls/'manifest.json')['dataset_sha256'],
         atlas_sha256=read(controls/'manifest.json')['frames'][0]['atlas_sha256'],
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
        'The ordinary camera gets the first decision on each source image. The experiment runs on its 118 abstentions. '
        'The free and boundary arms use scalar-frequency shortlists and independently checked 2-D phase gradients. '
        'The free arm sweeps possible horizons. The boundary arm limits the search to horizons near a long transition at both image borders.',
        '',
        'The source arm measures the repeated 16-texel motif directly. It rectifies the lower image at trial horizons, finds two independent '
        'brightness translations, and refines both across separated patches. A border transition adds hypotheses to a free grid. '
        'Every acceptance requires the ordinary same-sprite, orientation and origin check across both halves, and a metric camera lift. '
        'The repeat rule was frozen after training-source probes on Factory 01, Market Court 1, and Viaduct 01. Each newly supported camera was sent through the ordinary whole build and its exported ZIP was decoded. Recovered cameras were scored against '
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
    lines.extend(['', '## Split changes for the repeat-pattern fallback', '',
        '| Cohort | Split | Cameras recovered | Correct placements gained | Extra placements added | Strict F1 before → after | Strict position + ID F1 before → after |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for dataset in COHORTS:
        for split,after in datasets[dataset]['source']['splits'].items():
            before=datasets[dataset]['baseline']['splits'][split]
            g0,g1=before['geometry']['strict'],after['geometry']['strict']
            id0,id1=before['scores']['strict'],after['scores']['strict']
            lines.append(f"| {dataset} | {split} | {after['camera']['usable']-before['camera']['usable']} | "
                         f"{g1['correct']-g0['correct']} | {g1['extras']-g0['extras']} | "
                         f"{g0['f1']:.4f} → {g1['f1']:.4f} | {id0['f1']:.4f} → {id1['f1']:.4f} |")
    expected_passes=[r for r in diagnostics if r.get('reference_floor_sampler_accepted')]
    still_missing=[r for r in expected_passes if r['arms']['source']['camera']['P'] is None]
    reached_atlas=sum(r['arms']['source']['camera']['evidence']['tried']>0 for r in still_missing)
    reference_failures=[r for r in diagnostics if r.get('reference_floor_sampler_accepted') is False]
    correctly_abstained=sum(r['arms']['source']['camera']['P'] is None for r in reference_failures)
    pairs=[r['repeat_export_comparison'] for r in diagnostics if r.get('repeat_export_comparison')]
    identical=sum(r['identical_aligned_typed_placement'] for r in pairs)
    matrix_count=sum(r['identical_matrix'] for r in pairs)
    maximum=max(r['horizon_abs_delta_px'] for r in pairs)
    lines.extend(['', '## What the border measurement contributed', '',
        f'The free grid and the grid with border hypotheses supported the same {len(pairs)} views. '
        f'{matrix_count} had identical camera matrices. The largest difference in fitted horizon was {maximum:.3f} px. '
        f'Their decoded, translated block positions and IDs were identical in {identical} of {len(pairs)} recovered builds. '
        'See the repeat_no_boundary arm for the complete placement effect.',
        '', f'The recorded-camera atlas diagnostic had passed on {len(expected_passes)} of the 118 failed views. '
        f'The repeat-pattern arm recovered {len(expected_passes)-len(still_missing)} of those. In {reached_atlas} of {len(still_missing)} remaining views, '
        f'at least one metric chart reached bilateral registration but every candidate failed it. '
        f'Of {len(reference_failures)} views where the true chart failed registration, {correctly_abstained} stayed abstentions.',
        '', 'The two held-out Castle 01 views are sensitive to the evaluation origin near the horizon. '
        'Moving the evaluation-only integer ground registration by at most one block per horizontal axis reduced the 90th percentile corner error '
        'from 20.72 to 0.85 px on limestone and from 11.46 to 0.29 px on ashlar. All tables use the ordinary registration and placement scoring.',
        '', 'Keep this branch as an experiment. The repeat search recovered placements on held-out buildings in both expanded cohorts, '
        'which supports integrating it as an abstention-only fallback. The benchmark injects accepted cameras at the production handoff and '
        'does not install a default camera policy. The default branch has received no change from this experiment.',
        '', 'All original source-supported build exports were carried forward and verified by SHA-256. '
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
