"""Rerule supported masks, rerank native RGB and rerun the horizontal assembler."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_patch_combined as horizontal
from benchmark.floating_face_proposals import image_axes, source_offers
from benchmark.floating_patch_controls import inputs, ROOT, paths, read, save, digest, _native_alpha
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.relative_oracle_charts import fit_homography

GROUPS = ('original','fresh','supplemental','sky','holdout','holdout2')
OUT = ROOT/'runs/face-proposals'


def source_inputs(group, extra):
    if group != 'new6':
        return inputs(group)
    folder = Path(extra)
    return folder/'ordinary',folder/'ordinary/atlas/connected',folder/'source_material/charts',read(folder/'manifest.json')['cases']


def build(job):
    group,case,extra = job
    cv2.setNumThreads(1)
    src, ordinary, whole, _ = source_inputs(group, extra)
    name = case['id']
    rel = Path('sky')/name/'charts.json'
    handoff = src/'sky'/name/'source-handoff'
    image = Image.open(case['variants']['sky']['image']).convert('RGB')
    rgb = np.asarray(image)
    packet = read(whole/rel)
    paths.check_handoff(packet,handoff,digest(Path(case['variants']['sky']['image'])))
    seeds = {c['id']: c for c in read(ordinary/rel)['charts']}
    planes = {p['id']:p for p in read(handoff/'surfaces.json')['planes']}
    linear = image_axes(image,list(planes.values()))
    output, audit = [], []
    for chart in packet['charts']:
        parent = seeds[chart['id']]
        if not parent['cells'] or abs(chart['normal_camera'][1]) >= .67:
            output.append(chart)
            continue
        alpha, (x,y,x1,y1) = _native_alpha(planes[chart['id']]['support'],image.size)
        mask = np.zeros(rgb.shape[:2],bool)
        mask[y:y1,x:x1] = np.asarray(alpha)>0
        offers = source_offers(rgb,mask,chart['normal_camera'],linear)
        eligible = [(i,o) for i,o in enumerate(offers) if o['accept']]
        pick = max(eligible,key=lambda p:p[1]['score'])[0] if eligible else None
        replacement = chart
        if pick is not None:
            H = fit_homography([0,0], np.asarray(offers[pick]['quad'])[[3,2,1,0]])
            points = np.column_stack(np.nonzero(mask)[::-1])+.5
            measured = resample(chart,H,mask,rgb,points)
            replacement = preserve_and_rank(dict(parent,cells=[]),measured,rgb,mask)
            replacement['confidence'] = chart['confidence']
        audit.append(dict(id=chart['id'],picked=pick,offers=offers,
                          before=len(chart['cells']),after=len(replacement['cells'])))
        output.append(replacement)
    folder = OUT/group
    save(folder/'charts'/rel,dict(packet,charts=output,source_packet_sha256=digest(whole/rel)))
    save(folder/'offers'/(name+'.json.gz'),dict(source_sha256=packet['source_sha256'],charts=audit))
    with patch.object(horizontal,'inputs',lambda g:source_inputs(g,extra)):
        horizontal.run((group,case,folder/'charts',folder/'assembly'))
    return group,name,sum(row['picked'] is not None for row in audit)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('groups',nargs='+')
    cli.add_argument('--workers',type=int,default=5)
    cli.add_argument('--case',default='')
    cli.add_argument('--extra')
    cli.add_argument('--resume',action='store_true')
    cfg = cli.parse_args()
    jobs = [(g,c,cfg.extra) for g in cfg.groups for c in source_inputs(g,cfg.extra)[3] if cfg.case in c['id']]
    if cfg.resume:
        jobs = [j for j in jobs if not (OUT/j[0]/'assembly/connected-atlas/sky'/j[1]['id']/'score.json').is_file()]
    with ProcessPoolExecutor(cfg.workers) as pool:
        for job in as_completed([pool.submit(build,j) for j in jobs]):
            print(job.result(),flush=True)

