"""Diagnostic only: choose an offered plane depth using visible reference pixels.

One source per split, selected by lowest source SHA-256. Keep each source mask,
its estimated axis, ordinary depth offers, and all downstream geometry checks.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

from benchmark import concrete_shadow_boundaries, material_feedback_geometry as feedback
from benchmark import typed_structures as typed
import numpy as np
from benchmark.mask_blocks import registered
from benchmark.pavement_opening_eval import visible_faces
from benchmark.thin_face_placement_run import ref_inputs
from engine import atlas_plane_depth, foreground_conflicts, visible_placement
from engine.visible_raster import SAMPLE_STRIDE

ORIGINAL=foreground_conflicts.ForegroundConflicts


def reference(face, T, shift):
    normal=np.asarray(face['normal_world'])
    point=np.array(face['xyz_world'])+np.maximum(normal,0)
    local=T[:3,:3].T @ (point-shift-T[:3,3])
    direction=T[:3,:3].T @ normal
    axis=int(np.argmax(abs(direction)))
    return axis,int(round(local[axis])),int(round(direction[axis]))


def experiment(task):
    job,interventions=task
    split,frame,_,_=job
    key=frame['id']
    image,parts,counts,camera=typed.viewer.structure_stages(job,concrete_shadow_boundaries.baseline()[key])
    _,occupied,transform,_=ref_inputs(job)
    truth,ids=visible_faces(split,frame)
    labels={tuple(r['xyz_world']):r['block_id'] for r in truth['blocks']}
    from benchmark.strip_visibility_audit import horizontal_slivers
    optional=horizontal_slivers(truth,ids)
    shift=np.array(feedback.published()[1][key]['translation'])
    face_table=[None]+[reference(face,transform,shift) for face in truth['faces']]
    world_centers=np.array([f['xyz_world'] for f in truth['faces']])+.5
    local_centers=(world_centers-shift-transform[:3,3]) @ transform[:3,:3]
    face_cells=np.vstack((np.array([[-1000000]*3]),np.floor(local_centers).astype(int)))
    stride=SAMPLE_STRIDE
    native_slice=ids[stride//2::stride,stride//2::stride]
    lookup={(tuple(f['xyz_world']),int(np.argmax(abs(np.asarray(f['normal_world']))))):int(f['id'])
            for f in truth['faces']}
    cells=sorted(occupied)
    table=np.array([[lookup.get((cell,axis),0) for axis in range(3)] for cell in cells])
    W=np.asarray(truth['camera']['world_to_pixel_homogeneous'])
    eye=-np.linalg.solve(W[:,:3],W[:,3])
    def coarse_labels(offset):
        shifted=W.copy()
        shifted[:2]-=offset*W[2]
        owner,axes=visible_placement._frontmost_cells(cells,shifted,eye,native_slice.shape,face_axes=True)
        result=np.zeros(native_slice.shape,int)
        fg=owner>=0
        result[fg]=table[owner[fg],axes[fg]]
        return result,int((fg & (result==0)).sum())
    # Production rays land at 4k+2.0. A lookup in native [2::4] instead
    # selects native pixel centers at 4k+2.5, which can cross an edge.
    reduced,unmeasurable=coarse_labels(0.)
    check,unmeasurable_native=coarse_labels(.5)
    native_disagreements=int((check!=native_slice).sum())
    if native_disagreements or unmeasurable_native:
        raise ValueError(f'Native label raster crosscheck failed in {key}: {native_disagreements}, {unmeasurable_native}')
    alignment=dict(mismatched_old_face_ids=int((reduced!=native_slice).sum()),
                   ignored_new_subpixel_faces=unmeasurable,native_replay_disagreements=native_disagreements)
    def score(cells):
        positions={tuple(int(c[i]+shift[i]) for i in range(3)) for c in registered(cells,transform)}
        return typed.geometry_score(dict.fromkeys(positions),labels,optional)
    def run(oracle):
        calls=[]
        class Probe(ORIGINAL):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.audit={}
                calls.append(self.audit)

            def record(self,tag,mask,eye,rays,axis,baseline,rows,chosen,weights):
                super().record(tag,mask,eye,rays,axis,baseline,rows,chosen,weights)
                if tag not in self.planes:
                    return
                histogram=np.bincount(reduced[mask].ravel())
                planes=Counter()
                for i in np.flatnonzero(histogram[1:])+1:
                    planes[face_table[i]]+=int(histogram[i])
                strongest,n=planes.most_common(1)[0] if planes else ((-1,0,0),0)
                target_axis,target,side=strongest
                majority=2*n>mask.sum()
                eligible=bool(majority and target_axis==axis and np.sign(eye[axis]-target)==side)
                levels=[int(r['level']) for r in rows]
                self.audit[tag]=dict(axis=int(axis),surface=tag,mask_samples=int(mask.sum()),target_samples=n,
                    reference_axis=target_axis,reference_level=target,majority=bool(majority),eligible=eligible,
                    options=levels,joint_shortlist=list(map(int,self.planes[tag].options)),before_joint=int(chosen))
                if oracle=='pixel_positions':
                    offered_rays=rays[mask]
                    face_pixels=reduced[mask]
                    targets=face_cells[face_pixels]
                    # Weight each reference cube equally inside this mask. A large
                    # projected cube would otherwise dominate a distant row.
                    _, groups_, weights_=np.unique(targets,axis=0,return_inverse=True,return_counts=True)
                    pixel_weights=(face_pixels!=0)/weights_[groups_]
                    values={}
                    for candidate in levels:
                        distance=np.divide(candidate-eye[axis],offered_rays[:,axis],
                                           out=np.zeros(len(offered_rays)),where=np.abs(offered_rays[:,axis])>1.e-7)
                        projected=np.floor(eye+distance[:,None]*offered_rays).astype(int)
                        projected[:,axis]=candidate-int(eye[axis]>candidate)
                        exact=np.all(projected==targets,axis=1)&(distance>1.5)&(distance<150)&(projected[:,1]>=0)
                        values[candidate]=float(pixel_weights[exact].sum())
                    self.audit[tag]['pixel_scores']=values
            def resolve(self):
                levels=super().resolve()
                for tag,info in self.audit.items():
                    current=int(levels[tag])
                    target=info['reference_level']
                    info['ordinary']=current
                    available=target in info['options']
                    info['selected']=target if (oracle is True or tag==oracle) and info['eligible'] and available else current
                    if oracle=='pixel_positions':
                        scores=info.pop('pixel_scores')
                        best=max(scores,key=lambda c:(scores[c],c==current,-abs(c-current),-c))
                        info.update(selected=best,ray_score_before=scores.get(current,0.),ray_score_after=scores[best])
                    levels[tag]=info['selected']
                self.levels.update(levels)
                return levels
        stages=[]
        original_estimate=visible_placement._estimate_plane_cells
        original_place=atlas_plane_depth._place
        def candidates(*args,**kwargs):
            estimates=original_estimate(*args,**kwargs)
            stages.append({'raw_plane_candidates':score(set().union(*estimates))})
            return estimates
        def base(*args,**kwargs):
            result,status=original_place(*args,**kwargs)
            stages[-1]['after_frontmost_and_edges']=score(result)
            return result,status
        with patch.object(foreground_conflicts,'ForegroundConflicts',Probe),patch.object(
              visible_placement,'_estimate_plane_cells',candidates),patch.object(atlas_plane_depth,'_place',base):
            placed,status=atlas_plane_depth.place_ranked(image,'26.3',parts,counts,camera,atlas_path=typed.SUITE.ATLAS)
        return dict(passes=stages,ranked_placement=score(placed),planes=[list(c.values()) for c in calls],
                    roof_height_revisions=status.get('roof_height_revisions',{}))
    if interventions=='pixel_positions':
        return dict(id=key,source_hash=frame['files']['rgb.png'],oracle=run('pixel_positions'))
    if interventions is not None:
        return dict(id=key,individual=[dict(surface=tag, result=run(tag)['ranked_placement']) for tag in interventions])
    baseline=run(False); chosen=run(True)
    if baseline['planes'] and chosen['planes']:
        first=baseline['planes'][0]; second=chosen['planes'][0]
        assert [(r['surface'],r['options'],r['ordinary']) for r in first]==[(r['surface'],r['options'],r['ordinary']) for r in second]
    return dict(id=key,split=split,source_hash=frame['files']['rgb.png'],translation=shift.tolist(),ray_alignment=alignment,
                baseline=baseline,oracle=chosen,published_geometry=feedback.published()[1][key]['geometry'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-oracle',action='store_true')
    parser.add_argument('--individual',action='store_true')
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--splits',nargs='+',default=[*typed.SUITE.framed.SIZES,'shelter'])
    args=parser.parse_args()
    jobs=list(typed.SUITE.jobs(args.splits))
    selected=[min((j for j in jobs if j[0]==s),key=lambda j:j[1]['files']['rgb.png']) for s in args.splits]
    if args.ray_oracle:
        with ProcessPoolExecutor(args.workers) as pool:
            results=list(pool.map(experiment,[(job,'pixel_positions') for job in selected]))
        typed.compact(feedback.OUT/'plane-depth-ray-oracle.json',results)
        return
    if args.individual:
        existing=feedback.read(feedback.OUT/'plane-depth-cases.json')
        changed={r['id']:[p['surface'] for p in r['oracle']['planes'][0] if p['ordinary']!=p['selected']]
                 for r in existing if r['oracle']['planes']}
        tasks=[(job,[tag]) for job in selected for tag in changed.get(job[1]['id'],[])]
        with ProcessPoolExecutor(args.workers) as pool:
            comparisons=list(pool.map(experiment,tasks))
        typed.compact(feedback.OUT/'plane-depth-individual.json',comparisons)
        return
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(experiment,[(job,None) for job in selected]))
    def summarize(rows,member):
        allplanes=[r for view in rows for r in (view[member]['planes'] or [[]])[0]]
        eligible=[r for r in allplanes if r['eligible']]
        available=[r for r in eligible if r['reference_level'] in r['options']]
        counts=dict(surfaces=len(allplanes),majority=sum(r['majority'] for r in allplanes),
                    eligible=len(eligible),offered=len(available),
                    shortlist=sum(r['reference_level'] in r['joint_shortlist'] for r in available),
                    correct_ordinary=sum(r['reference_level']==r['ordinary'] for r in available))
        stage={}
        for name in ('raw_plane_candidates','after_frontmost_and_edges'):
            stage[name]=typed.pooled_scores([r[member]['passes'][-1][name] for r in rows if r[member]['passes']])
        stage['ranked_placement']=typed.pooled_scores([r[member]['ranked_placement'] for r in rows])
        return dict(availability=counts,stages=stage)
    summary={k:summarize(results,k) for k in ('baseline','oracle')}
    typed.compact(feedback.OUT/'plane-depth-cases.json',results)
    typed.compact(feedback.OUT/'plane-depth-summary.json',summary)


if __name__=='__main__':
    main()
