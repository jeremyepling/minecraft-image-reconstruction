"""Apply frozen gallery matchers to four additional oblique views of different buildings."""
import argparse
import json
from pathlib import Path
from benchmark.color_gallery import load, sample_labels
from engine.reconstruction import analyze, reconstruct


def evaluate(fixture, calibration, output, original=None):
    views = load(fixture/'views.json')
    blocks = load(fixture/'layout.json')['blocks']
    report = {}
    for family in ('canopy','stepped_tower','recess','corner'):
        split,angle=('test','front') if original else ('dev','left')
        view = next(v['id'] for v in views if v['split']==split and v['kind']==family and v['id'].endswith('-'+angle)
                    and (not original or (original/v['id']/'final-build/geometry.json').exists()))
        image = fixture/'captures'/f'{view}.png'
        out = output/view
        if original:
            grid=original/view/'final-build/geometry.json'
            cells=load(grid)['cells']
        else:
            _,_,cells = analyze(image,None,out,'texel')
            grid=out/'geometry.json'
        known = [b for b in blocks if b['building_id']==view.rsplit('-',1)[0]]
        truth,_ = sample_labels(cells,load(image.with_suffix('.json')),known)
        row = dict(crops=len(cells),clean=sum(t is not None for t in truth),methods={})
        for method in ('exposure','shared','motif','adaptive'):
            dest = out/method
            reconstruct(image,calibration/'atlas',None,calibration/'face_textures/softened-library.npz',
                        dest,grid=grid,match_exposure=method=='exposure',
                        match_appearance=None if method=='exposure' else method)
            placed = {tuple(b[k] for k in 'xyz'):b['block_id'] for b in load(dest/'placements.json')['blocks']}
            row['methods'][method] = sum(placed[tuple(c['xyz'])]==label for c,label in zip(cells,truth) if label)
        report[view] = row
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','calibration','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--original',type=Path,help='Use the four frozen front-view gallery grids instead')
    args=parser.parse_args()
    evaluate(args.fixture,args.calibration,args.out,args.original)
