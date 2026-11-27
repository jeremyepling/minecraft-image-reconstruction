"""Frozen silhouette-ruler input selection and separate source and scoring passes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from PIL import Image
from benchmark.expanded_baseline import ROOT,SOURCE,WORKTREES,read,save,digest
from benchmark.camera_grid_joint import BASE,GEO,MOGE,patches
from benchmark.camera_silhouette_ruler import outlines,orientation,runs,evaluate

RUN=ROOT/'runs/camera-silhouette-ruler-01'
REVIEW=ROOT/'benchmark/reviews/camera-silhouette-ruler-01'
CTRL=ROOT/'benchmark/reviews/camera-model-controls-01'


def one(row,stage):
    folder=SOURCE/row['dataset']/'data'/row['split']/'frames'/row['id']
    out=RUN/row['id']
    if stage=='source':
        rgb=folder/'rgb.png'
        base=BASE/row['dataset']/row['id']
        parts_path=base/'handoffs/placement-input.json.gz'
        parts=[p for p in read(parts_path)['parts'] if p.get('support')]
        geo_path=GEO/'proposals'/(row['id']+'.json')
        geo=read(geo_path)
        mo=MOGE/row['dataset']/row['id']
        proposal_path=mo/'proposal.json'
        proposal=read(proposal_path)
        assert digest(rgb)==geo['rgb_sha256']==proposal['source_sha256']==row['source_sha256']==read(base/'receipt.json')['source_sha256']
        scaffold=WORKTREES/'camera-structure-scaffold/runs/camera-structure-scaffold-02'/row['dataset']/row['id']/'source.json.gz'
        structural=read(scaffold)['structural_lines']
        native=np.array(Image.open(rgb).convert('RGB'))
        poly,masks=outlines(native,parts)
        L=orientation(geo,structural,native.shape[:2][::-1])
        traces=[] if L is None else runs(poly,L)
        anchors=patches(proposal.get('anchors',[]))
        with np.load(mo/'field.npz') as field:
            metric=evaluate(traces,poly,parts,masks,L,anchors,field,geo,native) if traces else []
        good=[m for m in metric if m['held']]
        # Abstain when multiple surviving integer scales for the same outline disagree.
        accept=good[0] if good and all(np.allclose(m['P'],good[0]['P'],atol=.01) for m in good) else None
        result=dict(id=row['id'],split=row['split'],hashes={str(p.relative_to(p.parents[1])):digest(p) for p in [rgb,scaffold,parts_path,geo_path,proposal_path,mo/'field.npz']},
                    polygons=[p.tolist() for p in poly],traces=traces,metric=metric,anchors=len(anchors),orientation=L is not None,
                    candidate=accept)
        save(out/'source.json.gz',result)
        return
    from benchmark.camera_silhouette_score import diagnose
    pred=read(out/'source.json.gz')
    labels=read(folder/'labels.json')
    d=diagnose([np.array(p) for p in pred['polygons']],pred['traces'],read(folder/'camera.json'),labels['faces'])
    save(out/'score.json.gz',d)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=['train','held'])
    parser.add_argument('stage',choices=['source','score'])
    parser.add_argument('--workers',type=int,default=4)
    a=parser.parse_args()
    rows=[r for r in read(CTRL/'independent-matched-cases.json.gz') if not r['camera_usable'] and (r['split']=='train')==(a.split=='train')]
    assert len(rows)==(49 if a.split=='train' else 21)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(partial(one,stage=a.stage),rows))
    print(a.stage,len(rows))


if __name__=='__main__':
    main()

