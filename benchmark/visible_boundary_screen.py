"""Freeze the source-only nomination before scoring any alternative."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.visible_boundary_capture import BASE, RUN, CASES, frames, COHORTS, SOURCE, configure, read, save
from benchmark.visible_boundary_proposal import nomination


def inputs(e):
    if e['dataset']==fixed.FIXED_DATASET:
        image, parts, _, camera=fixed.source_inputs(e)
    else:
        path=BASE/e['dataset']/e['id']
        camera=read(path/'build-manifest.json')['placement']
        image=Image.open(Path(e['frame'])/'rgb.png').convert('RGB')
        parts=read(RUN/e['dataset']/e['id']/'pre-inset.json.gz')
    return image,parts,camera


def task(entry):
    folder=RUN/entry['dataset']/entry['id']
    image,parts,camera=inputs(entry)
    event=nomination(image,parts,camera)
    parallel=nomination(image,parts,camera,'parallel')
    save(folder/'parallel-proposal.json',parallel)
    if parallel:
        print('parallel',entry['id'],parallel['id'],flush=True)
    save(folder/'proposal.json',event)
    return entry['id'],event and (event['id'],event['source'])


def entries():
    usable={(r['dataset'],r['id']) for r in read(CASES) if r['camera_usable']}
    return list(fixed.fixed_frames())+[e for e in frames(SOURCE,COHORTS) if (e['dataset'],e['id']) in usable]


if __name__=='__main__':
    configure()
    with ProcessPoolExecutor(3,initializer=configure) as pool:
        for a,b in pool.map(task,entries()):
            if b:
                print(a,b,flush=True)
