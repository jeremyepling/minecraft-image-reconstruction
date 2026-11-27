"""Evaluate the endpoint-only decision without the native atlas-depth veto."""
from benchmark.local_panel_depth import choose, materialize, trials, atlas
from engine.local_panel_depth import guard


def recover(image,parts,counts,camera,prior):
    scene,options=trials(image,parts,camera,prior)
    chosen=choose(options)
    if chosen is None:
        return prior,dict(options=len(options))
    keep,measurements=guard(scene,chosen,atlas())
    after=materialize(image,scene,prior,chosen) if keep else prior
    return after,dict(options=len(options),accepted=after!=prior,measurements=measurements)
