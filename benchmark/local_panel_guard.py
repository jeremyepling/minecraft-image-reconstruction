"""Require every exchanged source and destination to improve its own observed pixels."""
import numpy as np

from benchmark.local_panel_depth import trials, choose, materialize, render, Stack
from benchmark.wall_source_projection import FaceRenderer


def guard(scene,o):
    ids=[scene.cells.index(c) for c in o['before']]
    idx,owners,axes,score=render(scene,Stack(scene),o['before'],ids,o['after'],o['axis'])
    original=scene.owner[idx]
    old_axis=scene.axes[idx]
    inside=scene.union[idx]
    old_loss=scene.loss(idx,original,old_axis)
    new_loss=scene.loss(idx,owners,axes)
    renderer=FaceRenderer(scene)
    evidence=[]
    for j,cell in enumerate(o['after']):
        new=owners==len(scene.cells)+j
        old=original==ids[j]
        use=inside&renderer.test[idx]&renderer.fitted[idx]
        before=[]
        after=[]
        for a in np.unique(axes[new]):
            facet=new&(axes==a)
            fit=renderer.fit(idx[facet],cell,a)
            where=idx[facet&use]
            estimate,valid=renderer.draw(where,cell,a,fit)
            if valid and len(where):
                before.extend(abs(renderer.image[where]-renderer.source[where]).mean(1))
                after.extend(abs(estimate-renderer.source[where]).mean(1))
        evidence.append(dict(source_shape=float((old_loss[old]-new_loss[old]).sum()),
                             destination_shape=float((old_loss[new]-new_loss[new]).sum()),
                             texture_pixels=len(after),
                             texture=float(np.mean(before)-np.mean(after)) if after else None))
    # A local move has to explain the old pixels and improve both the geometry
    # and held-out texture at every destination. Ties remain ambiguous.
    accept=all(e['source_shape']>0 and e['destination_shape']>0 and
               e['texture_pixels']>=8 and e['texture']>0 for e in evidence)
    return accept,evidence


def recover(image,parts,counts,camera,prior):
    scene,options=trials(image,parts,camera,prior)
    chosen=choose(options)
    if chosen is None:
        return prior,dict(options=len(options))
    keep,measurements=guard(scene,chosen)
    after=materialize(image,scene,prior,chosen) if keep else prior
    return after,dict(options=len(options),accepted=after!=prior,measurements=measurements)
