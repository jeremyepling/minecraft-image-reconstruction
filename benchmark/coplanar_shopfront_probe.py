"""Isolate the Shopfront count change while keeping the two original masks separate."""
import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.coplanar_fragment_audit import CONTROL,REVIEW
from benchmark.coplanar_fragment_join import planes
from benchmark.coplanar_fragment_report import index
from benchmark.mask_blocks import decoded_pack,registered
from benchmark.texture_camera_eval import STABLE,checked,read,save
from engine.atlas_plane_depth import place_ranked
from engine.pillar_seams import native_mask,orientation
from engine.projected_appearance import Scene,STEP


def run():
    key='legacy-light_gray_concrete-dev-shopfront-plaster-low-left'
    s,fr,h=next(x for x in framed.cases(['regression']) if x[1]['id']==key)
    corpus={r['id']:r for r in read(REVIEW/'cases.json')}
    a,b=corpus[key]['thresholds']['standard']['groups'][0]['mask_ids']
    row=index(CONTROL)[key]
    image,parts,counts,camera=framed.photo(s,fr,h)
    truth,occupied,reg,_=framed.references(s,fr,None,framed.TRAINING_DATA,framed.DEPTH,{})
    pack=checked(CONTROL/s/key/'reconstruction.zip',row['pack_sha256'])
    prior=set(decoded_pack(pack))
    baseline=registered(prior,reg)
    scene=Scene(image,parts,camera,prior)
    p,rot=orientation(camera,image.size)
    inv=np.linalg.inv(p[:,:3]);eye=-inv@p[:,3]
    source={}
    for k in (a,b):
        part=next(part for part in parts if part['id']==k)
        mask=native_mask(part,image.size)
        axis=int(np.argmax(abs(rot.T@part['normal_camera'])))
        fit=planes(np.asarray(image),mask,axis,p,eye,inv,prior)
        m=mask[::STEP,::STEP].ravel()
        keep=m&(scene.owner>=0)&(scene.axes==axis)
        xyz=np.array(scene.cells)[scene.owner[keep]]
        levels,n=np.unique(xyz[:,axis],return_counts=True)
        source[k]=dict(pixel_samples=int(m.sum()),aligned=int(keep.sum()),
                       first_hit_levels=dict(zip(map(str,levels),n.tolist())),
                       texture_fit={c:fit[c] for c in ('level','margin','corr')},
                       original_counts=next(r['counts'] for r in counts['surfaces'] if r['id']==k))
    outcomes={}
    for tag,members in [('stem',[a]),('cap',[b]),('both',[a,b])]:
        rows=[dict(r,counts=[None,None],phase=[None,None],dimensions={'spans':[None,None],'blocks':[None,None]})
              if r['id'] in members else r for r in counts['surfaces']]
        found,_=place_ranked(image,'26.3',parts,dict(counts,surfaces=rows),camera,atlas_path=STABLE/'runs/26.3/atlas')
        scores=framed.placement_scores(registered(found,reg),baseline,dict.fromkeys(truth),occupied)
        outcomes[tag]=scores['relative_geometry']
    save(REVIEW/'shopfront-probe.json',dict(id=key,control_pack_sha256=row['pack_sha256'],fragments=source,
                                           counts_abstained_with_original_masks=outcomes))
    print(outcomes)


if __name__=='__main__':
    run()
