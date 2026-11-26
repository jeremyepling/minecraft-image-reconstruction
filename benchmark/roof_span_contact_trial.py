"""Test two levels of perimeter agreement on current placements in the training split."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

from benchmark import framed_depth_eval as data
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import ROOT, save
from engine.roof_span_contacts import recover


def score(job):
    split, f, row = job
    image, parts, counts, camera = data.photo(split, f, row)
    with ZipFile(data.REVIEW/'framed-depth-01/scores.zip') as archive:
        pack = archive.read(f"{split}/{f['id']}/reconstruction.zip")
    prior = set(decoded_pack(BytesIO(pack)))
    predictions = {mode:recover(image,parts,counts,camera,prior,mode=mode) for mode in ('guarded','broad')}
    truth, occupied, transform, _ = data.references(split,f,None,data.TRAINING_DATA,data.DEPTH,{})
    before = registered(prior,transform)
    results = {}
    for mode, cells in predictions.items():
        after = registered(cells,transform)
        result = placement_scores(after,before,dict.fromkeys(truth),occupied)['relative_geometry']
        results[mode] = {k:result[k] for k in ('gained','lost','extra_added','extra_removed','hidden_added')}
    return dict(id=f['id'],family=f['family'],pack_sha256=sha256(pack).hexdigest(),scores=results)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(score,data.cases(['train'])))
    assert len(rows)==72
    folder = ROOT/'runs/roof-span-contacts'
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'development.json',rows)
    for mode in ('guarded','broad'):
        print(mode,{k:sum(r['scores'][mode][k] for r in rows) for k in rows[0]['scores'][mode]})
