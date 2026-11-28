"""Propagate a freshly verified atlas face only through adjacent, photometrically compatible cells."""
from collections import deque
import numpy as np

from benchmark.floating_peer_recovery import descriptor, error, full_face, key, observed


def propagate(chart,mask,rgb,secondary=()):
    h=np.asarray(chart["H"])
    seeds=full_face(h,np.zeros(2),mask,rgb)
    if not seeds["ok"]:
        return chart,dict(accepted=False,proof=seeds)
    _,pix,coverage=observed(rgb,mask,h,np.zeros(2))
    reference=descriptor(pix[coverage])
    quarters=[descriptor(pix[y,x][coverage[y,x]]) for y in (slice(0,4),slice(4,9))
              for x in (slice(0,4),slice(4,9))]
    limit=float(np.clip(2.5*np.quantile([error(q,reference) for q in quarters],.9),7,15))
    supported=set()
    for cell in chart["cells"]:
        uv=key(cell)
        _,pix,visible=observed(rgb,mask,h,np.array(uv))
        if visible.mean()>=.78 and error(descriptor(pix[visible]),reference)<=limit:
            supported.add(uv)
    starts=({(0,0)} | set(secondary)) & supported
    queue=deque(starts)
    reached=set(starts)
    while queue:
        u,v=queue.popleft()
        for pos in ((u+1,v),(u-1,v),(u,v+1),(u,v-1)):
            if pos in supported and pos not in reached:
                queue.append(pos)
                reached.add(pos)
    cells=[dict(c,block_id=seeds["material"],fallback=False) if key(c) in reached else c for c in chart["cells"]]
    return dict(chart,cells=cells),dict(accepted=True,proof=seeds,threshold=limit,compatible=len(supported),
                                        reached=sorted(reached))
