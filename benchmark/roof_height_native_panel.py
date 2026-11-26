"""Score fixed-height roof choices and visible Minecraft placements on novel canopy geometries."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark import canopy_fixture_metrics as openings, relative_canopy_spacing as spacing
from benchmark import visible_depth_score as scorer, visible_depth_transfer as transfer
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from engine.surface_perimeter import _native_alpha
from benchmark.texture_camera_eval import ROOT, checked, digest, read, save
from benchmark.visible_recovery_report import summarize
from training.visible_labels import visible_labels


SELECTION=ROOT/'benchmark/reviews/roof-height-new-views/selection.json'


def decision_score(decision,parts,size,roof,target,camera):
    if not decision:
        return dict(part=None,block_y=None,exact=False,associated=False)
    part=next((p for p in parts if p['id']==decision['part']),None)
    if part is None:
        raise ValueError(f"Unknown predicted surface {decision['part']}")
    course=decision.get('block_y')
    if course is None:
        plane=decision['plane_y']
        P=np.asarray(camera['P'])
        eye=np.linalg.solve(P[:,:3],-P[:,3])
        course=plane-int(eye[1]>plane)
    pixels=_native_alpha(part['support'],size)
    count=overlap=0
    if pixels:
        alpha,(x0,y0,x1,y1)=pixels
        mask=np.asarray(alpha).astype(bool)
        count=int(mask.sum())
        overlap=int((roof[y0:y1,x0:x1]&mask).sum())
    return dict(part=part['id'],block_y=int(course),exact=course==target,
                associated=overlap>=16 and overlap>=.5*count,
                overlap_pixels=overlap,part_pixels=count)


def score(f, assets, data, sources, out, pose_prior=None):
    key=f['id']
    rgb=sources/f['directory']/'captures'/(f['source_id']+'.png')
    meta=f['files']
    base=data/'upstream'/key
    paths={name:checked(base/name,sha) for name,sha in assets.items()}
    am=read(paths['build-manifest.json'])
    placement=am['placement']
    assert am['image_sha256']==meta['rgb.png'] and am['through']=='placement'
    parts=read(paths['surfaces.json'])['planes']
    counts=read(paths['counts.json'])
    before=set(decoded_pack(paths['reconstruction.zip']))
    assert len(before)==placement['placed_blocks']
    camera=dict(P=placement['P'],scale_status=placement['camera_status'],reason=placement['reason'],
                camera_source=placement.get('camera_source'))
    if pose_prior is not None:
        folder,proof=pose_prior
        camera=read(checked(folder/key/'camera.json',proof['camera_sha256']))
        before=set(decoded_pack(checked(folder/key/'reconstruction.zip',proof['pack_sha256'])))
        paths['reconstruction.zip']=folder/key/'reconstruction.zip'
    supported=camera['P'] is not None
    with Image.open(checked(rgb,meta['rgb.png'])) as source:
        image=source.convert('RGB')
    result=scorer.predict(image,parts,counts,camera,before.copy()) if scorer.predict else before
    cells,status=result if isinstance(result,tuple) else (result,{})
    dest=out/key
    dest.mkdir(parents=True,exist_ok=True)
    pack=dest/'reconstruction.zip'
    if cells==before:
        pack.write_bytes(paths['reconstruction.zip'].read_bytes())
    else:
        _write_read(pack,cells)
    after=set(decoded_pack(pack))
    assert after==cells and (supported or not (before or after))
    # Open recorded camera and world only after the single-image export.
    pose=read(checked(rgb.with_suffix('.json'),meta['camera.json']))
    raw=read(checked(rgb.with_name(f['source_id']+'-blocks.json'),meta['world-readback.json']))
    states={tuple(b[k] for k in 'xyz'):b for b in raw}
    truth,ids,_=visible_labels(pose,states,(0,0,0))
    visible={tuple(b['xyz_world']):b['block_id'] for b in truth['blocks']}
    occupied=transfer.structure(raw,64)
    roof_world=max(y for x,y,z in occupied)
    floor=min(y for x,y,z in occupied)
    front_z=max(z for x,y,z in occupied if y==floor)
    front=sorted(x for x,y,z in occupied if y==floor and z==front_z)
    assert len(front)>=2 and front[1]>front[0]+1 and floor==65 and roof_world-floor==f['target_block_y']
    # Score the leftmost front bay on both structures; the high roof has three front posts.
    spec=dict(left=front[0]+1,right=front[1],front_z=front_z,bottom=floor,roof_y=roof_world)
    roof=np.isin(ids,[face['id'] for face in truth['faces'] if face['xyz_world'][1]==roof_world])
    transform=registration(np.asarray(camera['P']),pose) if supported else None
    after_world=registered(after,transform)
    before_world=registered(before,transform)
    target=roof_world-floor
    if supported:
        assert target==roof_world-transform[1,3]
    primary=decision_score(status.get('primary_height'),parts,image.size,roof,target,camera)
    accepted=[decision_score(d,parts,image.size,roof,target,camera)
              for d in status.get('height_decisions',[]) if d.get('accepted')]
    elevated=Counter(y for x,y,z in before if y>0)
    modal=max(elevated,key=lambda y:(elevated[y],y)) if elevated else None
    geometry=placement_scores(after_world,before_world,visible,occupied.keys())
    geometry['relative_geometry'].pop('joint_correct')
    roof_result=spacing.score(after_world,set(visible),spec)
    with patch.object(transfer,'BUILDS',sources):
        rays=openings.score(f,after_world,spec)['rays']
    save(dest/'cells.json',dict(local=sorted(after),world=sorted(after_world)))
    return dict(id=key,shape=f['shape'],rgb_sha256=meta['rgb.png'],reference_hashes=meta,input_hashes=assets,
                pack_sha256=digest(pack),camera_supported=supported,camera_source=camera['camera_source'],
                expected_course=target,prior_modal=modal,prior_tallest=max(elevated,default=None),
                primary=primary,accepted=accepted,status=status,roof=roof_result,clear_rays=len(rays),
                filled_clear_rays=sum(r['foreground_filled'] for r in rays),**geometry)


def aggregate(rows):
    return dict(views=len(rows),camera_supported=sum(r['camera_supported'] for r in rows),
                prior_modal_exact=sum(r['prior_modal']==r['expected_course'] for r in rows),
                prior_tallest_exact=sum(r['prior_tallest']==r['expected_course'] for r in rows),
                primary_exact_and_on_roof=sum(r['primary']['exact'] and r['primary']['associated'] for r in rows),
                primary_exact=sum(r['primary']['exact'] for r in rows),
                primary_wrong=sum(r['primary']['block_y'] is not None and not r['primary']['exact'] for r in rows),
                primary_missing=sum(r['primary']['block_y'] is None for r in rows),
                anchored=sum(r['roof']['anchored_roof_cells'] for r in rows),
                visible_roof=sum(r['roof']['visible_roof_cells'] for r in rows),
                widths_exact=sum(r['roof']['observed_gap_width']==r['roof']['target_gap_width'] for r in rows),
                widths_unresolved=sum(r['roof']['observed_gap_width'] is None for r in rows),
                clear_rays=sum(r['clear_rays'] for r in rows),filled_clear_rays=sum(r['filled_clear_rays'] for r in rows),
                geometry=summarize(rows))


def run(args):
    data=args.data
    sources=args.sources or (data/'sources' if (data/'sources').is_dir() else transfer.BUILDS)
    manifest=read(data/'builds.json')
    frames=read(checked(args.selection,manifest['selection_sha256']))['selected']
    builds={r['id']:r['files'] for r in manifest['rows']}
    assert builds.keys()=={f['id'] for f in frames}
    root=args.candidate_root.resolve()
    scorer.init_candidate(root,args.candidate)
    pose_priors={}
    if args.pose_diagnostic:
        pose_metadata=read(args.pose_diagnostic/'index.json')
        assert pose_metadata['builds_sha256']==digest(data/'builds.json')
        pose_priors={r['id']:(args.pose_diagnostic,r) for r in pose_metadata['rows']}
        assert pose_priors.keys()==builds.keys()
    rows=[score(f,builds[f['id']],data,sources,args.out,pose_priors.get(f['id'])) for f in frames]
    groups=dict(all=aggregate(rows),supported=aggregate([r for r in rows if r['camera_supported']]))
    groups.update({f'height_{h}':aggregate([r for r in rows if r['expected_course']==h])
                   for h in sorted({r['expected_course'] for r in rows})})
    save(args.out/'cases.json',rows)
    summary=dict(method=args.candidate,control='54c9fee',control_build_revision=manifest['revision'],
                 camera_mode='supplied-pose diagnostic' if args.pose_diagnostic else 'ordinary photograph only',
                 candidate_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 selection_sha256=digest(args.selection),input_manifest_sha256=digest(data/'builds.json'),
                 cases_sha256=digest(args.out/'cases.json'),groups=groups)
    if args.candidate:
        summary['adapter_sha256']=digest(root.joinpath(*args.candidate.partition(':')[0].split('.')).with_suffix('.py'))
    if args.pose_diagnostic:
        summary['pose_prior_sha256']=digest(args.pose_diagnostic/'index.json')
    save(args.out/'summary.json',summary)
    print(groups['all'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--selection',type=Path,default=SELECTION)
    p.add_argument('--sources',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,default=ROOT)
    p.add_argument('--candidate')
    p.add_argument('--pose-diagnostic',type=Path,help='Folder of frozen recorded-pose diagnostic priors')
    run(p.parse_args())
