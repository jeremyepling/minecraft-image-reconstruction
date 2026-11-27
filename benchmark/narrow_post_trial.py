"""Place individually supported visible rows on distinct photographed pillar planes."""
import numpy as np

from benchmark.narrow_post_rows import predict
from benchmark.wall_texture_masks import native_mask,orientation,rewrite
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def placed(image,parts,counts,camera,prior):
    if not prior or not camera.get('P') or not parts:
        return prior,{}
    regions,ops=rewrite(image,parts,camera,'pillar')
    if not ops:
        return prior,{}
    p,_=orientation(camera,image.size)
    eye=-np.linalg.solve(p[:,:3],p[:,3])
    native=p.copy()
    native[:2]*=SAMPLE_STRIDE
    size=image.size[::-1]
    union=np.logical_or.reduce([native_mask(part,image.size) for part in parts])
    by_id={piece['id']:piece for piece in regions}
    supported={}
    for op in ops:
        children=[by_id[op['ids'][0]+'-tex-'+s] for s in ('0','1')]
        plans=[predict(image,piece,camera,prior) for piece in children]
        if any(q is None or q['score']<.65 or q['margin']<.18 for q in plans):
            continue
        if plans[0]['level']==plans[1]['level'] or plans[0]['axis']!=plans[1]['axis']:
            continue
        for region,q in zip(children,plans):
            mask=native_mask(region,image.size)
            values,n=np.unique(q['ijk'],axis=0,return_counts=True)
            for xyz in values[n>=24]:
                cell=tuple(map(int,xyz))
                if cell[1]>=0 and cell not in prior:
                    supported[cell]=(mask,q['axis'])
    proposed=set(supported)
    baseline=_frontmost_cells(sorted(prior),native,eye,size) if proposed else None
    kept=set(proposed)
    while kept:
        sequence=sorted(prior|kept)
        owner,axes=_frontmost_cells(sequence,native,eye,size,face_axes=True)
        index={c:i for i,c in enumerate(sequence)}
        result=set()
        for c in kept:
            mask,axis=supported[c]
            visible=owner==index[c]
            face=visible & (axes==axis)
            if (face & mask).sum()<24 or (face & mask).sum()<.75*face.sum():
                continue
            if (visible & ~union & (baseline<0)).any():
                continue
            result.add(c)
        result=_keep_prior_visible(image.size,p,eye,prior,result)-prior if result else set()
        if kept==result:
            break
        kept=result
    return prior|kept,dict(proposed=len(proposed),added=len(kept))
