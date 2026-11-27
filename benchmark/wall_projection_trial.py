"""Image-only one-block depth moves using an estimated-camera render of both alternatives."""
from benchmark.wall_source_projection import Scene, FaceRenderer, measure_moves
from benchmark.wall_projection_move_review import winner
from engine.visible_horizontal_faces import _keep_prior_visible


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
