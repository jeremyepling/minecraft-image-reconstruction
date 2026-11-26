"""Score genuinely visible downward faces in the 80 fixed held/regression images."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark.canopy_fixture_metrics import raster
from benchmark.texture_camera_eval import DATA, checked, digest, frames, read, save


def score(job):
    f, outputs = job
    folder = DATA/'data'/f['split']/f['directory']
    labels = read(checked(folder/'labels.json', f['files']['labels.json']))['faces']
    down = {tuple(face['xyz_world']) for face in labels if face['direction'] == 'down'}
    cells_file = outputs/f['split']/f['id']/'cells.json'
    cells = set(map(tuple,read(cells_file)['world']))
    common = cells & down
    found = set()
    if common:
        pose = read(checked(folder/'camera.json', f['files']['camera.json']))
        visible, _ = raster(pose,cells)
        found = {tuple(face['xyz_world']) for face in visible if face['direction'] == 'down'} & down
    return dict(id=f['id'], split=f['split'], target=len(down), coordinate_present=len(common),
                recovered_face=len(found), cells_sha256=digest(cells_file))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    a = p.parse_args()
    cases = [r for split in ('test','validation','regression') for r in frames(DATA,split)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(score,[(r,a.predictions) for r in cases]))
    summary = {}
    for split in sorted({r['split'] for r in rows}):
        totals = Counter()
        for row in rows:
            if row['split'] == split:
                totals.update({k:row[k] for k in ('target','coordinate_present','recovered_face')})
        summary[split] = dict(totals)
    save(a.out,dict(summary=summary,cases=rows))
    print(summary)
