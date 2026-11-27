"""Label competing image-only inset hypotheses after all predictions for that view are frozen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import facade_inset_round as shared
from benchmark.texture_camera_eval import digest, save
from engine.pillar_seams import native_mask


def collect(job):
    split, frame, prior, control = job
    image, parts, _, camera = shared.ordinary_inputs(split, frame, prior, control)
    proposals = shared.refine(image, parts, camera)
    samples = []
    if proposals:
        labels, truth = shared.visible_map(split, frame, metadata=True)
        masks = {p['id']:p for p in parts}
        best = shared.surface_stats(parts, truth, image.size)
        by_surface = {f['surface_id']:f for f in labels['faces']}
        hosts = {}
        areas = np.bincount(truth.ravel())
        for p in proposals:
            x,y,r,b = p['roi_xyxy']
            pixels = truth[y:b,x:r][p['mask']]
            tally = np.bincount(pixels, minlength=len(areas))
            sid = int(tally.argmax())
            if p['host'] not in hosts:
                hm = native_mask(masks[p['host']], image.size)
                hosts[p['host']] = int(np.bincount(truth[hm], minlength=len(areas)).argmax())
            host = hosts[p['host']]
            iou = float(tally[sid]/max(1, areas[sid]+len(pixels)-tally[sid])) if sid else 0.
            purity = float(tally[sid]/max(1,len(pixels))) if sid else 0.
            offset = None
            if sid and host:
                a, c = by_surface[host], by_surface[sid]
                if a['normal_world'] == c['normal_world']:
                    offset = int(np.dot(a['normal_world'], np.subtract(a['xyz_world'], c['xyz_world'])))
            novel = sid > 0 and sid not in best['matched']
            good = bool(novel and purity >= .8 and iou >= .5 and offset == p['features']['depth'] and offset > 0)
            samples.append(dict(group=[p[k] for k in ('host','side','sill')],rim=p['rim'],
                features={k:float(v) for k,v in p['features'].items()},strict=p['strict'],
                grade=dict(correct=good, reference=sid, host_surface=host, actual_depth=offset,
                           novel=bool(novel), iou=round(iou,6), purity=round(purity,6))))
    return dict(id=frame['id'],split=split,rgb_sha256=frame['files']['rgb.png'],
                family=frame.get('family') or control.get('family') or frame.get('shape'),
                layout=frame.get('layout') or frame.get('shape') or frame['id'], samples=samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root',type=Path,default=Path('../facade-inset-returns'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=shared.initialize,
            initargs=(args.candidate_root.resolve(),'benchmark.inset_return_proposals:proposals')) as pool:
        rows=list(pool.map(collect,shared.jobs([*shared.framed.SIZES,'shelter'])))
    args.out.mkdir(exist_ok=True,parents=True)
    save(args.out/'rows.json',rows)
    print(len(rows),'views',sum(len(r['samples']) for r in rows),'hypotheses',digest(args.out/'rows.json'))


if __name__=='__main__':
    main()
