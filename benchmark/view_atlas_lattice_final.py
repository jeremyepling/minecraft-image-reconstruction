"""Apply the exact standalone point/footprint policies to original and revised grid geometry."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
from io import BytesIO
from zipfile import ZipFile
import argparse
import numpy as np
from PIL import Image
from benchmark import view_atlas_lattice_run as stage,full_block_lattice_run as release
from benchmark.view_conditioned_atlas_match import rerank
from benchmark.expanded_baseline import read,save,digest
from benchmark.full_block_lattice import line_camera,atlas
from benchmark.mask_blocks import decoded_pack


def source(entry):
    key,ds=entry['id'],entry['dataset']
    if ds==release.FIXED:
        image=Image.open(entry['rgb']).convert('RGB')
        status=release.fixed._jobs()[key][3]['status']
        with ZipFile(release.typed.REVIEW/'typed-exports.zip') as archive:
            pack=archive.read(key+'/reconstruction.zip')
        inventory={tuple(r[:3]):r[4] for r in release.sites()[key]['rows']}
        folder=None
    else:
        folder=release.BASE/ds/key
        image=Image.open(Path(entry['frame'])/'rgb.png').convert('RGB')
        status=read(folder/'build-manifest.json')['placement']
        pack=(folder/'reconstruction.zip').read_bytes()
        inventory={tuple(r[c] for c in 'xyz'):r['material_source'] for r in read(folder/'placements.json')['blocks']}
    return image,status,pack,inventory,folder


def selected(result_folder,status,baseline,folder):
    row=read(result_folder/'result.json')
    evidence=read(result_folder/'evidence.json.gz')
    payload=(result_folder/'proposal.zip').read_bytes() if row['changed'] else baseline
    camera=dict(status)
    if row['rescued']:
        P=line_camera(read(folder/'handoffs/mlsd.json.gz')['segments'],(1440,900))
        eye=np.array(evidence['eye'])
        camera.update(camera_status='supported',P=np.column_stack((P,-P@eye)).tolist(),
                      camera_source='structure-lattice',reason=evidence['reason'])
    return row,payload,camera


def measure(entry):
    key,ds=entry['id'],entry['dataset']
    root=stage.RUN/ds/key
    out=root/'final'
    out.mkdir(exist_ok=True)
    image,status,baseline,inventory,folder=source(entry)
    from hashlib import sha256
    receipt=dict(source_sha256=entry['rgb_sha256'],baseline_pack_sha256=sha256(baseline).hexdigest(),id=key,dataset=ds,policies={})
    data={}
    for arm,work in [('grid',stage.OLD/ds/key),('joint',root)]:
        row,payload,camera=selected(work,status,baseline,folder)
        save(out/(arm+'-camera.json'),camera)
        source_name=arm if arm=='grid' else 'joint_anchor'
        destinations={p:name.removeprefix('minecraft:') for p,name in decoded_pack(BytesIO(payload)).items()}
        evidence=inventory|{p:'fallback' if name=='smooth_stone' else 'source-atlas'
                            for p,name in destinations.items() if p not in inventory}
        assert evidence.keys()==destinations.keys()
        (out/(source_name+'.zip')).write_bytes(payload)
        data[source_name]=(payload,camera,evidence)
        static,view,details=rerank(image,camera,destinations,atlas())
        eligible=len(details)
        for name,ids in [(arm+'_point',static),(arm+'_view',view)]:
            updates={p for p in ids if ids[p]!=destinations[p]}
            pack=release.typed.typed_pack(ids)[0] if updates else payload
            (out/(name+'.zip')).write_bytes(pack)
            methods=evidence|{p:'source-atlas:view-second-look' for p in updates}
            data[name]=(pack,camera,methods)
            receipt['policies'][name]=dict(eligible=eligible,changed=len(updates),participating=len(ids),
                                          fallback=sum(v=='fallback' for v in methods.values()))
        save(out/(arm+'-eligibility.json.gz'),details)
        receipt['policies'][source_name]=dict(eligible=eligible,changed=0,participating=len(destinations),
                                              fallback=sum(v=='fallback' for v in evidence.values()))
    # Each material arm is evaluated after both inference passes finish.
    if ds==release.FIXED:
        ref=release.fixed.fixed_reference(entry)
        truth={tuple(r['xyz_world']):r['block_id'] for r in ref[0]['blocks']}
        partial=set()
        def grade(data):
            payload,cam,assignment=data
            return release.fixed.score_result(entry,release.fixed.ReplayResult(payload,assignment,cam),reference=ref)
    else:
        ref=release.scorer.load_reference(entry)
        truth=ref.truth; partial=set().union(*(ps for kind,ps in ref.by_kind.items() if kind!='full_cube'))
        def grade(data):
            payload,cam,assignment=data
            return release.scorer.evaluate_pack(entry,payload,cam,assignment=assignment,reference=ref)
    data['baseline']=(baseline,status,inventory)
    maps={}; metrics={}; seen={}
    for name,(payload,cam,assignment) in data.items():
        # Comparative roles may produce byte-identical packs. Their fallback and camera fields are part of the key.
        fingerprint=(sha256(payload).hexdigest(),tuple(map(tuple,cam['P'])) if cam.get('P') else (),tuple(sorted(p for p,v in assignment.items() if v=='fallback')))
        if fingerprint not in seen:seen[fingerprint]=grade((payload,cam,assignment))
        row,mapping=seen[fingerprint]
        receipt['policies'].setdefault(name,{})['sha256']=sha256(payload).hexdigest()
        metrics[name]=dict(release.compact_row(row),registered=row['registration']['geometry'],registered_scores=row['registration']['scores'])
        maps[name]=mapping
    assert metrics['grid']['geometry']==read(stage.OLD/ds/key/'result.json')['source']['geometry']
    assert metrics['joint_anchor']['geometry']==read(root/'result.json')['source']['geometry']
    for name in data:
        if name!='baseline':
            methods=data[name][2]
            save(out/(name+'-assignments.json.gz'),[[*p,v] for p,v in sorted(methods.items())])
    save(out/'metrics.json',dict(metrics=metrics,
        against_grid=release.compare(maps['grid'],maps['joint_view'],truth,partial),
        against_baseline=release.compare(maps['baseline'],maps['joint_view'],truth,partial)))
    save(out/'receipt.json',receipt)
    return ds,key


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--ids',nargs='*')
    a=p.parse_args()
    todo=[e for e in stage.all_frames() if (not a.ids or e['id'] in a.ids)
          and (stage.RUN/e['dataset']/e['id']/'result.json').exists()
          and not (stage.RUN/e['dataset']/e['id']/'final/receipt.json').exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(measure,todo):print(r,flush=True)


if __name__=='__main__':
    main()
