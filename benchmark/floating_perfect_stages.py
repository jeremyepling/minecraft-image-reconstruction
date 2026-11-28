"""Reference-only stage interventions on the original V1 Floating benchmark."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import cv2
import numpy as np
from PIL import Image, ImageFilter

from benchmark import floating_v1_expanded as expanded, floating_v1_run as v1
from benchmark import floating_pregrid_depth_refit as pregrid
from benchmark.camera_labels import screen_matrix
from benchmark.expanded_baseline import ROOT, WORKTREES, read, save, digest
from benchmark.expanded_wall_source_resample import TEXELS, atlas, rank_tile
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_oracle_charts import fit_homography, unit_projection, UNIT
from engine import floating_sky
from engine.geometry import project
from engine.mask_first import _image_depth
from engine.normal_regions import _field
from engine.plane_consensus import with_mask
from engine.surface_perimeter import _native_alpha

INCUMBENT=WORKTREES/'floating-visible-builds-v1'
REFERENCE=WORKTREES/'floating-perfect-stage-report/runs/v1-release'
FROZEN=INCUMBENT/'runs/floating-v1'
FIELDS=INCUMBENT/'runs/v1-fields'
OUT=ROOT/'runs/perfect-stages'
REVIEW=ROOT/'benchmark/reviews/floating-perfect-stage-inputs-01'
V1_ARCHIVE=ROOT/'benchmark/reviews/floating-visible-builds-v1-01'
BASE_REV='01d51c0'
ARMS=('surface-mask','surface-plane','wall-grid')


def fingerprint(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fixture(case):
    frame=Path(case['frame'])
    if not (frame/'labels.npz').exists():
        release=Path(os.getenv('FLOATING_V1_RELEASE',REFERENCE))
        options=(release/'data'/case['split']/'frames'/case['id'],release/case['split']/'frames'/case['id'],
                 release/'frames'/case['id'])
        frame=next((p for p in options if (p/'labels.npz').is_file()),options[0])
    for name,expected in {**case['reference_sha256'],'rgb.png':case['rgb_sha256']}.items():
        assert digest(frame/name)==expected,(case['id'],name)
    return dict(case,frame=str(frame),variants={'sky':dict(image=str(frame/'rgb.png'),source_sha256=case['rgb_sha256'])})


def native(case):
    frame=Path(case['frame'])
    labels=read(frame/'labels.json')
    pose=read(frame/'camera.json')
    with np.load(frame/'labels.npz') as raw:
        owner=raw['face_id']
    pixel_counts=np.bincount(owner.ravel().astype(int))
    assert np.count_nonzero(pixel_counts[1:])==len(labels['faces'])
    ids=np.zeros(len(pixel_counts),'int32')
    camera=screen_matrix(pose)
    origin=np.asarray(pose['eye_world'])
    errors=[]
    groups=defaultdict(list)
    for face in labels['faces']:
        fid=face['id']
        assert pixel_counts[fid]==face['visible_pixels']>0
        sid=face['surface_id']
        ids[fid]=sid
        corners=np.asarray(face['corners_world'])
        homo=(corners-origin)@camera[:,:3].T+camera[:,3]
        assert (homo[:,2]>0).all()
        errors.append(np.max(np.linalg.norm(homo[:,:2]/homo[:,2:]-face['corners_px'],axis=1)))
        groups[sid].append(face)
    assert max(errors)<1.e-4
    return owner,ids[owner],groups,labels,pose,camera,float(max(errors))


def mask_inputs(case,native_data,metric,with_quads=True):
    owner,surface,groups,labels,pose,cam,_=native_data
    path=FIELDS/case['id']
    size=owner.shape[::-1]
    with np.load(path/'normal.npz') as f:
        field=_field(f['value'],size)
    with np.load(path/'depth.npz') as f:
        depth=_image_depth(f['value'],size)
    image=Image.open(case['variants']['sky']['image']).convert('L')
    gray=np.asarray(image.filter(ImageFilter.GaussianBlur(.9)),dtype='f')[...,None]
    basis=cam[:,:3].copy()
    basis[:2]-=np.array(size)[:,None]/2*basis[2]
    basis/=np.linalg.norm(basis,axis=1)[:,None]
    inverse=np.linalg.inv(cam[:,:3])
    eye=np.asarray(pose['eye_world'])
    center=eye-inverse@cam[:,3]
    parts=[]
    plane_depth={}
    for sid,faces in groups.items():
        region=surface==sid
        normal=np.median(field[region],axis=0)
        normal/=max(np.linalg.norm(normal),1.e-8)
        relative=float(np.median(depth[region]))
        first=faces[0]
        if metric:
            n=np.asarray(first['normal_world'])
            distance=np.asarray(first['corners_world'])[0]@n
            assert all(np.array_equal(f['normal_world'],n) and np.allclose(np.asarray(f['corners_world'])@n,distance) for f in faces)
            y,x=np.nonzero(region)
            uv=np.column_stack((x+.5,y+.5,np.ones(len(x))))
            z=(distance-center@n)/((uv@inverse.T)@n)
            assert np.min(z)>0
            normal=basis@n
            relative=float(np.median(1/z))
            plane_depth[sid]=relative
        plane=dict(id='visible-'+str(sid),normal_camera=normal.tolist(),relative_inverse_depth=relative)
        if with_quads:
            part=with_mask(plane,region,size,gray,np.ones(2))
            alpha,(x,y,r,b)=_native_alpha(part['support'],size)
            assert np.array_equal(np.asarray(alpha)>0,region[y:b,x:r]) and part['support_area']==int(region.sum())
        else:
            part=plane
        parts.append(part)
    if metric:
        # Exact median optical inverse depth is kept in blocks^-1, and an explicitly
        # documented scene-relative monotonic scaling adapts the existing [0,1] field.
        lo,hi=min(plane_depth.values()),max(plane_depth.values())
        for part in parts:
            value=part['relative_inverse_depth']
            part['measured_inverse_depth_blocks']=value
            part['relative_inverse_depth']=(value-lo)/max(hi-lo,1.e-12)
    if with_quads:
        assert sum(p['support_area'] for p in parts)==np.count_nonzero(owner)
        parts.sort(key=lambda p:-p['support_area'])
    return parts


def wall_charts(native_data,parts,rgb):
    owner,_,groups,labels,_,_,_=native_data
    kinds={v['id']:v.get('geometry_kind','full_cube') for v in labels['blocks']}
    normals={p['id']:p['normal_camera'] for p in parts}
    charts=[]
    audit=Counter()
    for sid,faces in groups.items():
        if faces[0]['normal_world'][1]:
            continue
        wall=[f for f in faces if kinds[f['block']]=='full_cube']
        if not wall:
            continue
        axis=next(i for i,n in enumerate(wall[0]['normal_world']) if n)
        axes=tuple(k for k in range(3) if k!=axis)
        at=lambda f: np.asarray(f['xyz_world'])[list(axes)]
        origin=np.min([at(f) for f in wall],axis=0)
        primary=max(wall,key=lambda f:f['visible_pixels'])
        h=fit_homography(at(primary)-origin,unit_projection(primary,axes))
        cid='visible-'+str(sid)
        cells=[]
        for f in wall:
            uv=at(f)-origin
            err=float(np.max(np.linalg.norm(project(h,UNIT+uv)-unit_projection(f,axes),axis=1)))
            assert err<1.e-4,(sid,err)
            audit['max_grid_error_px']=max(audit['max_grid_error_px'],err)
            px=np.floor(project(h,TEXELS+uv)).astype(int)
            x,y=px[...,0],px[...,1]
            inside=(x>=0)&(x<owner.shape[1])&(y>=0)&(y<owner.shape[0])
            x,y=x.clip(0,owner.shape[1]-1),y.clip(0,owner.shape[0]-1)
            visible=inside&(owner[y,x]==f['id'])
            ranked=rank_tile(dict(rgb=rgb[y,x],mask=visible),atlas())
            block=ranked['block_id']
            audit['atlas_fallback']+=int(not block)
            cells.append(dict(u=int(uv[0]),v=int(uv[1]),block_id=block or 'smooth_stone',fallback=not bool(block),
                              confidence=max(0.,ranked['evidence'].get('correlation',0.))))
        assert len(set((c['u'],c['v']) for c in cells))==len(cells)
        charts.append(dict(id=cid,H=h.tolist(),normal_camera=normals[cid],cells=cells,confidence=1.))
        audit['visible_vertical_units']+=len(cells)
    return charts,dict(audit)


def score(case,folder):
    receipt=read(folder/'score.json')
    local=decoded_pack(folder/'reconstruction.zip')
    coords=read(folder/'placements.json')
    assert local=={tuple(p):m for p,m in coords['local']}
    assert digest(folder/'reconstruction.zip')==receipt['pack_sha256']
    assert receipt['reference_sha256']==case['reference_sha256']
    fixed=receipt['score']['fixed']
    measured=fixed or receipt['score']['relaxed_yaw']
    out={}
    for metric in ('geometry','scores'):
        for mode in ('strict','facade'):
            st=measured[metric][mode]
            c,m,e=(st[k] for k in ('correct','missed','extras'))
            if local and fixed is None:
                c,m,e=0,c+m,len(local)
            out[metric+'_'+mode]=dict(correct=c,missed=m,extras=e,f1=2*c/max(1,2*c+m+e))
    return dict(metrics=out,pack_sha256=receipt['pack_sha256'],charted=receipt['charts_seen'],
                unresolved=bool(local) and fixed is None,components=receipt['component_sizes'],selected=receipt['selected_origins']),(
                {tuple(p):m for p,m in coords['aligned']} if fixed else {})


def downstream_signature():
    # Hash the entire tracked Python consumer surface. An unrelated code change can
    # conservatively invalidate a result; changing any used consumer cannot reuse one.
    paths=sorted(p for d in ('engine','benchmark','calibration') for p in (ROOT/d).rglob('*.py') if p.name!='floating_perfect_stages.py')
    extra=sorted(p for p in (ROOT/'benchmark/data/count-atlas-v1').rglob('*') if p.is_file())
    return fingerprint([(p.relative_to(ROOT).as_posix(),digest(p)) for p in paths+extra])


def run(job):
    arm,case,downstream=job
    cv2.setNumThreads(1)
    case=fixture(case);name=case['id'];rel=Path('sky')/name/'charts.json'
    existing=read(FROZEN/'rows'/(name+'.json.gz'))
    base=Path(existing['roots']['nonshrinking'])/v1.PREFIX/name
    fields=read(FIELDS/name/'receipt.json')
    v1.require_fields(case,FIELDS/name,fields)
    config=dict(arm=arm,source_sha256=case['rgb_sha256'],references=case['reference_sha256'],
                comparison_arm='current',comparison_sha256=digest(base/'reconstruction.zip'),
                baseline_score_sha256=digest(base/'score.json'),baseline_chart_sha256=read(base/'score.json')['chart_sha256'],
                stage_source_sha256=digest(Path(__file__)),consumers_sha256=downstream,base_revision=BASE_REV,
                fields_sha256=fields['fields_sha256'],models=fields['model_sha256'],model_settings=fields['preprocessing'],
                atlas=atlas().provenance,interpreter=os.sys.version,
                versions={k:version(k) for k in ('numpy','Pillow','opencv-python')},mode='conservative-v1-current')
    key=fingerprint(config)
    out=OUT/'cache'/key
    rec_file=out/'result.json'
    if rec_file.is_file():
        receipt=read(rec_file)
        assert all(digest(out/p)==h for p,h in receipt['artifacts'].items())
        return receipt
    native_data=native(case)
    truth={tuple(b['xyz_world']):b['block_id'] for b in native_data[3]['blocks']}
    image=Image.open(case['variants']['sky']['image']).convert('RGB')
    parts=mask_inputs(case,native_data,arm=='surface-plane',with_quads=arm!='wall-grid')
    origin=out/'pipeline'
    if os.name=='nt':
        origin=Path('\\\\?\\'+str(origin.resolve()))
    src=origin/'source';g='v1'
    out.mkdir(parents=True,exist_ok=True)
    save(out/'input.json.gz',dict(id=name,arm=arm,planes=parts))
    estimates={}
    if arm=='wall-grid':
        # The chart input does not contain world axes, inter-chart origins, reference
        # materials or truth joins. The frame gauges are independently zeroed.
        raw=read(FROZEN/'split-border/charts'/rel)
        charts,estimates=wall_charts(native_data,parts,np.asarray(image))
        handoff=origin/'charts'/rel
        save(handoff,dict(raw,charts=charts))
        with patch.object(v1.assembly,'load_reference',v1.score_reference),patch.object(pregrid,'data',lambda g:(FROZEN/'source',)):
            pregrid.replay(g,case,origin/'charts',origin/'assembly',{})
        folder=origin/'assembly'/v1.PREFIX/name
        save(out/'input.json.gz',dict(id=name,arm=arm,charts=charts))
    else:
        def inject(_,proposals):
            return parts,dict(intervention=arm,kept_count=len(parts),input_count=len(proposals))
        with ExitStack() as ctx:
            for attr,val in (('OUT',origin),('SOURCE',src),('FIELDS',FIELDS)):
                ctx.enter_context(patch.object(v1,attr,val))
            ctx.enter_context(patch.object(floating_sky,'filter_proposals',inject))
            v1.predict(case)
            expanded.run(case)
        row=read(origin/'rows'/(name+'.json.gz'))
        folder=Path(row['roots']['nonshrinking'])/v1.PREFIX/name
    base_stats,prev=score(case,base)
    stats,curr=score(case,folder)
    a,b=set(prev),set(curr);ids0={p for p in a if prev[p]==truth.get(p)};ids1={p for p in b if curr[p]==truth.get(p)}
    paired=dict(gained=len((b-a)&truth.keys()),lost=len((a-b)&truth.keys()),exact_gained=len(ids1-ids0),exact_lost=len(ids0-ids1))
    # Unknown-orientation outputs are charged as extras by the policy, with no
    # registered coordinate attribution. Keep their counts in metrics.
    for f in ('reconstruction.zip','placements.json','score.json'):
        (out/f).write_bytes((folder/f).read_bytes())
    artifacts={p:digest(out/p) for p in ('input.json.gz','reconstruction.zip','placements.json','score.json')}
    result=dict(id=name,split=case['split'],stage=arm,cache_key=key,config=config,base=base_stats,output=stats,paired=paired,
                corner_error_px=native_data[-1],frontmost_pixels=int(np.count_nonzero(native_data[0])),
                native_surface_count=len(native_data[2]),grid=estimates,artifacts=artifacts)
    save(rec_file,result)
    print(arm,name,stats['metrics']['geometry_strict'],flush=True)
    return result


def publish(rows,stage):
    REVIEW.mkdir(parents=True,exist_ok=True)
    save(REVIEW/(stage+'-conditions.json.gz'),rows)
    cohorts={'all':rows,**{split:[r for r in rows if r['split']==split] for split in sorted({r['split'] for r in rows})}}
    summary={}
    for k,records in cohorts.items():
        kinds={}
        for arm in ('base','output'):
            kinds[arm]={}
            for field in records[0][arm]['metrics']:
                totals=Counter()
                for r in records:
                    totals.update({m:r[arm]['metrics'][field][m] for m in ('correct','missed','extras')})
                c,m,e=(totals[l] for l in ('correct','missed','extras'))
                kinds[arm][field]=dict(totals,f1=2*c/max(1,2*c+m+e))
        summary[k]=dict(cases=len(records),**kinds)
    save(REVIEW/(stage+'-summary.json'),summary)
    with ZipFile(REVIEW/(stage+'-inputs-and-exports.zip'),'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for r in rows:
            folder=OUT/'cache'/r['cache_key']
            for filename in r['artifacts']:
                info=ZipInfo(r['id']+'/'+filename,date_time=(1980,1,1,0,0,0))
                info.compress_type=ZIP_DEFLATED
                archive.writestr(info,(folder/filename).read_bytes())


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--stage',choices=ARMS,required=True)
    cli.add_argument('--split',default='test')
    cli.add_argument('--workers',type=int,default=4)
    args=cli.parse_args()
    sig=downstream_signature()
    cases=[c for c in v1.tasks() if args.split=='all' or c['split']==args.split]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(run,[(args.stage,c,sig) for c in cases]))
    publish(rows,args.stage)
