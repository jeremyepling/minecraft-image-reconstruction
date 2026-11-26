"""Validate complete decisions on the training set using family-out weights per image."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np

from benchmark.section_decision_fit import fit, sample
from benchmark.plane_section_trial import jobs, source, RUN, _write_read, registered, placement_scores
from benchmark.texture_camera_eval import save
from engine import section_decision as policy


def replay(job):
    task,model,cut=job
    split,f,*_=task
    with np.load(RUN/'evidence'/split/(f['id']+'.npz'),allow_pickle=False) as z:
        data=(z['points'],z['owner'],z['P'],z['eye'],z['rows'].tolist()) if 'points' in z else None
    image,parts,camera,prior=source.load_prediction(task)
    obs=policy.observations(data)
    decisions=policy.choose(obs,model,*cut)
    result=policy.apply(image,prior,data,decisions)
    path=RUN/'held-family'/f['id']
    path.mkdir(parents=True,exist_ok=True)
    local=_write_read(path/'reconstruction.zip',result)
    T,truth,occupied=source.labels(task,camera['P'])
    return dict(id=f['id'],family=f['family'],**placement_scores(registered(local,T),registered(prior,T),truth,occupied))


if __name__=='__main__':
    tasks=jobs('train')
    current=json.loads(policy.WEIGHTS.read_bytes())
    cut=current['add_threshold'],current['remove_threshold']
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(sample,tasks))
        models={group:fit([r for r in rows if r[1]!=group]) for group in sorted({r[1] for r in rows})}
        results=list(pool.map(replay,[(t,models[t[1]['family']],cut) for t in tasks]))
    save(RUN/'held-applied.json',results)
    sums={group:sum((Counter(r['relative_geometry']) for r in results if r['family']==group),Counter()) for group in models}
    print({g:{k:c[k] for k in ('gained','lost','extra_added','extra_removed')} for g,c in sums.items()})
