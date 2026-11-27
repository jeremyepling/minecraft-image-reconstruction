"""Keep original wall cells that still have separate photographed support beside the moved panel."""
import numpy as np

from benchmark.local_panel_depth import choose, trials
from benchmark.wall_source_projection import Scene
from engine.visible_horizontal_faces import _keep_prior_visible


def apply(image,parts,camera,prior,scene,option):
    if option is None:
        return prior,{}
    sources=set(option['before'])
    dest=set(option['after'])
    both=Scene(image,parts,camera,prior|dest)
    ids={c:i for i,c in enumerate(both.cells)}
    def witness(c):
        pixels=both.owned.get(ids[c],())
        if not len(pixels):
            return 0
        return int(np.count_nonzero(both.union[pixels] & both.masks[both.axes[pixels],pixels]))
    kept={c for c in sources if witness(c)>=4}
    # Original and proposed tiles have separate direct source witnesses in a
    # single render. Reject proposed moves that require erasing an observable
    # original tile just to make the destination visible.
    new_support=[witness(c) for c in sorted(dest)]
    if min(new_support)<4:
        return prior,dict(reason='occluded',surviving_sources=len(kept),destination_pixels=new_support)
    rest=(prior-sources)|kept
    after=_keep_prior_visible(image.size,scene.P,scene.eye,rest,dest)
    if after!=rest|dest:
        return prior,dict(reason='intrudes')
    return after,dict(reason='independent first hits',surviving_sources=len(kept),destination_pixels=new_support)


def recover(image,parts,counts,camera,prior):
    scene,options=trials(image,parts,camera,prior)
    return apply(image,parts,camera,prior,scene,choose(options))
