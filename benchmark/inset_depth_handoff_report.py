"""Re-score altered images against identical surface masks with the depth hint disabled."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import benchmark
from benchmark import framed_depth_eval as f
from benchmark.mask_blocks import registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import ROOT,checked,digest,read,save
from benchmark.wall_texture_evaluate import metrics
from engine import atlas_plane_depth, surface_model, texture_guided_count

benchmark.__path__.append(str(ROOT.parent/'facade-inset-round/benchmark'))
from benchmark import facade_inset_round as shared


def compare(job):
    split,frame,prior,control,row,root=job
    key=frame['id']
    im,_,counts,cam=shared.ordinary_inputs(split,frame,prior,control)
    parts=read(checked(root/split/key/'surfaces.json',row['surfaces_sha256']))['planes']
    plain=[{k:v for k,v in p.items() if k!='inset_back_plane'} for p in parts]
    atlas=None
    if 'atlas_phase' in counts:
        from engine.atlas_count import load_atlas
        atlas=load_atlas(shared.ATLAS,'26.3')
    with TemporaryDirectory() as tmp:
        measured=surface_model._count_regions(im,parts,'26.3',texture_guided_count.METHOD,atlas,Path(tmp))
    dims={r['id']:r['dimensions']['blocks'] for r in measured['surfaces']}
    cells0,status0=atlas_plane_depth.place_ranked(im,'26.3',plain,measured,cam,atlas_path=shared.ATLAS)
    cells1,status1=atlas_plane_depth.place_ranked(im,'26.3',parts,measured,cam,atlas_path=shared.ATLAS)
    assert split!='shelter'
    ref='old' if split in ('extra','new') else split
    builds=f.NEW_INPUTS if split=='new' else f.transfer.BUILDS
    truth,occupied,T,_=f.references(ref,frame,f.NATIVE_INPUTS.get(split),f.TRAINING_DATA,f.DEPTH,
                            dict(native=f.native.BUILDS,old=builds,fresh=builds))
    before,after=registered(cells0,T),registered(cells1,T)
    score=placement_scores(after,before,dict.fromkeys(truth),occupied)
    labels,ref_pixels=shared.visible_map(split,frame,metadata=True)
    by_part={p['id']:p for p in metrics(parts,ref_pixels,im.size)['parts']}
    faces={p['surface_id']:p for p in labels['faces']}
    def plane(face):
        axis=int(np.argmax(abs(np.array(face['normal_world']))))
        return axis,int(face['xyz_world'][axis]) + int(face['normal_world'][axis]>0)
    old_levels={r['id']:r['level'] for r in status0['plane_offsets']}
    new_levels={r['id']:r['level'] for r in status1['plane_offsets']}
    records=[]
    for p in parts:
        if not (fit:=p.get('inset_back_plane')):
            continue
        axis=fit['axis']
        target=faces[by_part[p['id']]['dominant']]
        host=faces[by_part[fit['host']]['dominant']]
        a,pos=plane(target)
        ah,host_pos=plane(host)
        world_axis=int(np.argmax(abs(T[:3,axis])))
        sign=int(T[world_axis,axis])
        records.append(dict(id=p['id'],target=target['surface_id'],dimensions=dims[p['id']],
            old_level=old_levels[p['id']],new_level=new_levels[p['id']],front=fit['front'],
            measured_relative_depth=fit['signed_depth'],true_relative_depth=(pos-host_pos)*sign if a==ah==world_axis else None,
            correct_relative_depth=a==ah==world_axis and (pos-host_pos)*sign==fit['signed_depth'],
            world_plane_before=sign*old_levels[p['id']]+int(T[world_axis,3]),
            world_plane_after=sign*new_levels[p['id']]+int(T[world_axis,3]),truth_world_plane=pos))
    result=score['relative_geometry']
    result.pop('joint_correct',None)
    assert all(result[k]==row['geometry'][k] for k in ('correct','extras','hidden_extras'))
    return dict(id=key,split=split,rgb_sha256=frame['files']['rgb.png'],surfaces_sha256=row['surfaces_sha256'],
                compared=records, geometry=result, changed_sorted_cells=cells0!=cells1,
                other_levels_changed=[k for k in old_levels if old_levels[k]!=new_levels[k]
                                      and k not in {m['id'] for m in records}])


def run(source,out,workers):
    out.mkdir(parents=True,exist_ok=True)
    report=read(checked(source/'cases.json',read(source/'summary.json')['cases_sha256']))
    rows={r['id']:r for r in report if r['changed']}
    jobs=[(*j,rows[j[1]['id']],source) for j in shared.jobs([*f.SIZES,'shelter']) if j[1]['id'] in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cases=list(pool.map(compare,jobs))
    save(out/'comparison.json',dict(views=len(report),inactive=len(report)-len(rows),
                                    source_sha256=digest(source/'cases.json'),cases=cases))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',type=Path,default=Path('runs/handoff'))
    ap.add_argument('--out',type=Path,default=Path('benchmark/reviews/inset-depth-handoff-01'))
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    run(args.runs,args.out,args.workers)
