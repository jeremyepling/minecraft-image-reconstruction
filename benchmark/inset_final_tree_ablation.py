"""Run the complete final gate twice, changing only whether the learned decision is consulted."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256

import numpy as np

from benchmark.inset_mask_chooser_eval import OUT, inspect, shipped, suite
from benchmark.inset_mask_chooser import refine
from benchmark.inset_outline_count_guard import apply
from benchmark.texture_camera_eval import digest, read, save
from engine import inset_returns
from engine.facade_inset_phase import supplemental
from engine.pillar_seams import native_mask


def signatures(image, parts, events):
    used={e['output'] for e in events}|{e['host'] for e in events}
    return {p['id']:sha256(np.packbits(native_mask(p,image.size)).tobytes()).hexdigest() for p in parts if p['id'] in used}


def compare(job):
    split,frame,_,_=job
    image,original,counts,camera=suite.ordinary_inputs(*job)
    found,strict=inset_returns.refine(image,original,camera) if inset_returns.possible(image.size,original) else (original,[])
    before,texture=supplemental(image,found,camera,strict) if strict else (found,[])
    frozen=shipped()[frame['id']]
    assert frame['files']['rgb.png']==frozen['rgb_sha256'] and (len(strict),len(texture))==(frozen['returns'],frozen['texture'])
    outputs=[]
    prior=[]
    final=[]
    atlas=None
    for tree in (True,False):
        tentative,pre=refine(image,original,camera,before,strict,learned=tree)
        if pre and 'atlas_phase' in counts and atlas is None:
            from engine.atlas_count import load_atlas
            atlas=load_atlas(suite.ATLAS,'26.3')
        parts,kept=apply(image,before,tentative,pre,camera,atlas,collar=True)
        prior.append(len(pre))
        final.append(kept)
        outputs.append(parts)
    same_ids=[p['id'] for p in outputs[0]]==[p['id'] for p in outputs[1]]
    same_footprints=same_ids and all(a is b or np.array_equal(native_mask(a,image.size),native_mask(b,image.size))
                                      for a,b in zip(*outputs))
    canonical=lambda ee:[(e['host'],e['output'],e['side'],e['sill'],e['photo_grid_span']) for e in ee]
    a,b=canonical(final[0]),canonical(final[1])
    row=dict(id=frame['id'],split=split,rgb=frame['files']['rgb.png'],prior_filter=prior,final=[len(a),len(b)],
             identical_surfaces=same_footprints,identical_counts_and_anchors=a==b)
    if a or b:
        row.update(anchors_with_tree=a,anchors_without_tree=b,hashes_with_tree=signatures(image,outputs[0],final[0]),
                   hashes_without_tree=signatures(image,outputs[1],final[1]))
    if not same_footprints or a!=b:
        row['with_tree']=inspect((True,True,True,job))
        row['without_tree']=inspect((False,True,True,job))
    return row


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=7) as pool:
        rows=list(pool.map(compare,suite.jobs([*suite.framed.SIZES,'shelter'])))
    ref=read(OUT/'collar.json')
    assert len(rows)==len(ref['frames'])==204
    assert all((r['split'],r['id'],r['rgb'],r['final'][0])==(v['split'],v['id'],v['rgb'],v['accepted'])
               for r,v in zip(rows,ref['frames']))
    groups={'development':['train'],'fixed':suite.framed.EVALUATION,'alternate':['extra'],'new':['new'],'shelter':['shelter']}
    summary={}
    for name,splits in groups.items():
        checked=[r for r in rows if r['split'] in splits]
        summary[name]=dict(views=len(checked),tree_final=sum(r['final'][0] for r in checked),
            no_tree_final=sum(r['final'][1] for r in checked),tree_before_count=sum(r['prior_filter'][0] for r in checked),
            no_tree_before_count=sum(r['prior_filter'][1] for r in checked),identical_footprints=sum(r['identical_surfaces'] for r in checked),
            identical_counts_and_anchors=sum(r['identical_counts_and_anchors'] for r in checked))
    save(OUT/'final-tree-ablation.json',dict(reference_sha256=digest(OUT/'collar.json'),model_sha256=digest(OUT/'model.json'),
        groups=summary,frames=rows))
    print(summary)
