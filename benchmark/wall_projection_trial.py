"""Image-only one-block depth moves using an estimated-camera render of both alternatives."""
from benchmark.wall_source_projection import Scene, FaceRenderer, measure_moves
from benchmark.wall_projection_move_review import winner
from engine.visible_horizontal_faces import _keep_prior_visible
import numpy as np


def proposals(image,parts,camera,prior):
    if not prior or not parts or camera.get('P') is None:
        return [],None
    scene=Scene(image,parts,camera,prior)
    renderer=FaceRenderer(scene)
    return measure_moves(scene,renderer,minimum_geometry=0),scene


def apply(method,trials,scene,image,prior):
    decision=winner(trials,method)
    if decision is None:
        return prior,dict(method=method,abstained=True)
    old,to=decision['old'],decision['to']
    result=_keep_prior_visible(image.size,scene.P,scene.eye,prior-{old},{to})
    if to not in result:
        return prior,dict(method=method,abstained=True,veto=True)
    return result,dict(method=method,decision=decision)


def both(image,parts,counts,camera,prior):
    trials,scene=proposals(image,parts,camera,prior)
    return apply('both',trials,scene,image,prior)


def necessary_deltas(scene,n,to):
    """Test changing each half of a move while holding the other half in place."""
    ii,aa,zz=scene.faces(to)
    idx=np.union1d(scene.owned[n],ii)
    original=scene.owner[idx]
    old_axis=scene.axes[idx]
    without=original.copy()
    without_axis=old_axis.copy()
    without_z=scene.dep[idx].copy()
    removal=original==n
    was=idx[removal]
    without[removal],without_axis[removal]=scene.back[was],scene.back_axis[was]
    without_z[removal]=scene.back_depth[was]
    pix=np.searchsorted(idx,ii)
    # Scene A retains the old block as well as the proposed one. Scene B removes
    # the old without inserting a new block. Compare both to the actual move.
    both,both_axis=original.copy(),old_axis.copy()
    near=zz>scene.dep[ii]+1.e-10
    both[pix[near]],both_axis[pix[near]]=n,aa[near]
    moved,moved_axis=without.copy(),without_axis.copy()
    near=zz>without_z[pix]+1.e-10
    moved[pix[near]],moved_axis[pix[near]]=n,aa[near]
    final_loss=scene.loss(idx,moved,moved_axis)
    return dict(deletion_evidence=round(float((scene.loss(idx,both,both_axis)-final_loss).sum()),2),
                addition_evidence=round(float((scene.loss(idx,without,without_axis)-final_loss).sum()),2))


def both_necessary(image,parts,counts,camera,prior):
    trials,scene=proposals(image,parts,camera,prior)
    result,status=apply('both',trials,scene,image,prior)
    if result==prior:
        return prior,status
    d=status['decision']
    evidence=necessary_deltas(scene,scene.cells.index(d['old']),d['to'])
    status.update(evidence)
    if min(evidence.values())<2:
        status['veto']='both endpoints need separately observable support'
        return prior,status
    return result,status
