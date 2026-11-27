"""Paired component bypasses on the same ten image-selected ranked placement inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from inspect import getsource
import sys
from unittest.mock import patch

from benchmark import concrete_shadow_boundaries, material_feedback_geometry as exp, typed_structures as typed
from benchmark.pavement_opening_eval import visible_faces
from benchmark.relative_geometry import best_translation
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import ref_inputs
from engine import atlas_plane_depth, foreground_conflicts, placed_materials, plane_repeat_evidence, visible_placement

MODES=('control','skip_repeat_choice','skip_joint','skip_occlusion','skip_edges')


def independent(policy):
    levels={tag:plane.base for tag,plane in policy.planes.items()}
    policy.levels.update(levels)
    return levels


def bypass(mode):
    stack=ExitStack()
    if mode=='half_pixel_primary_rays':
        # Benchmark-only, revision-pinned one-line substitution. Leave native
        # texel fits, recorded labels, calibrated camera and visibility raster alone.
        original=visible_placement._estimate_plane_cells
        body=getsource(original)
        old='[(xx + .5) * SAMPLE_STRIDE, (yy + .5) * SAMPLE_STRIDE, np.ones(shape)]'
        new='[(xx + .5) * SAMPLE_STRIDE + .5, (yy + .5) * SAMPLE_STRIDE + .5, np.ones(shape)]'
        if body.count(old)!=1:
            raise ValueError('Primary sample centers changed; inspect this ablation before replay')
        namespace=vars(visible_placement).copy()
        exec(compile(body.replace(old,new),'<primary-half-pixel-trial>','exec'),namespace)
        stack.enter_context(patch.object(visible_placement,original.__name__,namespace[original.__name__]))
    elif mode=='skip_repeat_choice':
        stack.enter_context(patch.object(plane_repeat_evidence,'select_repeat',lambda rows,base,weights:base))
    elif mode=='skip_joint':
        stack.enter_context(patch.object(foreground_conflicts.ForegroundConflicts,'resolve',independent))
    elif mode in ('skip_occlusion','skip_edges'):
        original=atlas_plane_depth._place
        def skip(*args,**kwargs):
            kwargs['remove_occluded' if mode=='skip_occlusion' else 'recover_edges']=False
            return original(*args,**kwargs)
        stack.enter_context(patch.object(atlas_plane_depth,'_place',skip))
    elif mode!='control':
        raise ValueError(mode)
    return stack


def metrics(ids,evidence,transform,truth,optional):
    payload,decoded=typed.typed_pack(ids)
    local=list(decoded)
    coords=typed.aligned_sites(local,transform,(0,0,0))
    shift,_=best_translation(set(coords),set(truth)) if local else ((0,0,0),0)
    aligned={c:tuple(p[i]+shift[i] for i in range(3)) for c,p in zip(local,coords,strict=True)}
    predictions={aligned[c]:name for c,name in decoded.items()}
    assigned={aligned[c] for c,s in evidence.items() if s!='fallback'}
    return dict(pack_sha256=sha256(payload).hexdigest(),translation=list(shift),
                geometry=typed.geometry_score(predictions,truth,optional),
                result=typed.score(predictions,truth,optional,assigned)),predictions


def run_job(task):
    job,modes=task
    split,frame,_,_=job
    key=frame['id']
    image,parts,counts,camera=typed.viewer.structure_stages(job,concrete_shadow_boundaries.baseline()[key])
    truth_labels,pixels=visible_faces(split,frame)
    truth={tuple(b['xyz_world']):b['block_id'] for b in truth_labels['blocks']}
    optional=horizontal_slivers(truth_labels,pixels)
    _,_,T,_=ref_inputs(job)
    atlas=typed.viewer.calibrated_atlas()
    cases={}; baseline=None;old_cells=None;old_ids=None;old_evidence=None
    for mode in modes:
        with bypass(mode):
            cells,status=atlas_plane_depth.place_ranked(image,'26.3',parts,counts,camera,atlas_path=typed.SUITE.ATLAS)
        # Identical input to the deterministic material stage can share its result.
        if old_cells is not None and cells==old_cells:
            ids,evidence=old_ids,old_evidence
        else:
            ids,evidence=placed_materials.assign(image,camera,cells,atlas)
        scores,pred=metrics(ids,evidence,T,truth,optional)
        got=set(pred)&set(truth); extra=set(pred)-set(truth)
        exact={p for p in got if pred[p]==truth[p]}
        if mode=='control':
            baseline=(got,extra,exact)
            old_cells,old_ids,old_evidence=cells,ids,evidence
        normal,wrong,typed_correct=baseline
        paired=dict(gained=len(got-normal),lost=len(normal-got),extra_added=len(extra-wrong),
                    extra_removed=len(wrong-extra),typed_gained=len(exact-typed_correct),typed_lost=len(typed_correct-exact))
        cases[mode]=dict(scores,relative_to_control=paired,joint_reassigned=status['foreground_depth'].get('changed_planes',0),
                         rejected_occluded=status.get('fully_occluded_blocks',0),verified_edges=status.get('verified_edge_blocks',0))
    return dict(id=key,split=split,source_sha256=frame['files']['rgb.png'],methods=cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',action='store_true')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--splits',nargs='+',default=[*typed.SUITE.framed.SIZES,'shelter'])
    args=parser.parse_args()
    jobs=list(typed.SUITE.jobs(args.splits))
    selected=[min((j for j in jobs if j[0]==s),key=lambda j:j[1]['files']['rgb.png']) for s in args.splits]
    modes=('control','half_pixel_primary_rays') if args.calibration else MODES
    suffix='primary-ray-calibration' if args.calibration else 'component-bypass'
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(run_job,[(j,modes) for j in selected]))
    typed.compact(exp.OUT/(suffix+'-cases.json'),results)
    summary={}
    for mode in modes:
        current=[r['methods'][mode] for r in results]
        paired={k:sum(r['relative_to_control'][k] for r in current) for k in current[0]['relative_to_control']}
        summary[mode]=dict(geometry=typed.pooled_scores([r['geometry'] for r in current]),
                           result=typed.pooled_scores([r['result'] for r in current]),paired=paired)
    typed.compact(exp.OUT/(suffix+'-summary.json'),summary)


if __name__=='__main__':
    if len(sys.argv)>3 and sys.argv[1]=='--ordinary':
        # The standard wrapper supplied its normal environment and build arguments.
        import reconstruct
        mode=sys.argv[2]
        sys.argv=sys.argv[3:]
        with bypass(mode):
            reconstruct.main()
    else:
        main()
