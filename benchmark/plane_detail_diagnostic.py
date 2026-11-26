"""Evaluation-only best overlap of eligible recorded surfaces with each stage's regions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import read, save, digest
from benchmark.crop_fit_round import BASE
from benchmark.normal_round import groups
from benchmark.plane_detail import register, source_field
from benchmark.surface_detection_maps import _intersections, _native_reference, _ratios
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.strip_neighbor import clean_surfaces
from runtime.mode import image_only, mark_image_only


def evaluate(job):
    case,row,sha,tokens = job
    with Image.open(checked(case.path('image'),row['sha256'])) as opened:
        image = opened.convert('RGB')
    parts,_ = clean_surfaces(image,read_gzip(checked(BASE/row['predictions'],sha))['gap'])
    pose,ref = recorded(case)
    owners = _native_reference(pose,ref,read(case.path('labels'))['faces'])
    count = max(ref['patches'])+1
    sizes = np.bincount(owners.ravel(),minlength=count)
    eligible = [k for k,v in ref['patches'].items() if v['eligible']]
    _,_,full = _intersections(parts,ref,owners)
    stages = {'shipped_final': _ratios(full,sizes).max(axis=0)[eligible].tolist()}
    for token in tokens:
        raw = source_field(row['sha256'],token)
        for stride in (1,2,4):
            registered = register(raw,image.size,stride)
            hits = np.bincount((registered.astype(np.int64)*count+owners).ravel(),
                minlength=(int(raw.max())+1)*count).reshape(-1,count)
            stages[f'tokens{token}_stride{stride}'] = _ratios(hits[1:],sizes).max(axis=0)[eligible].tolist()
    return dict(key=row['key'],sha256=row['sha256'],cohort=row['cohort'],reference_ids=eligible,stages=stages)


@image_only()
def run(split, tokens, out):
    groups_by_split,_,_ = groups()
    cases = {c.floor+'/'+c.id:c for g in groups_by_split.values() for c in g}
    manifest = read(BASE/'manifest.json')
    rows = read(checked(BASE/'cases.json',manifest['artifacts']['cases.json']))
    jobs = [(cases[r['key']],r,manifest['artifacts'][r['predictions']],tokens) for r in rows
            if split=='all' or r['cohort']==split]
    with ProcessPoolExecutor(max_workers=5,initializer=mark_image_only) as pool:
        records = list(pool.map(evaluate,jobs))
    summary = {}
    for group in ('all',*sorted({r['cohort'] for r in records})):
        summary[group] = {}
        for stage in records[0]['stages']:
            values = np.asarray([v for record in records if group in ('all',record['cohort']) for v in record['stages'][stage]])
            summary[group][stage] = dict(eligible=len(values),best_iou=float(values.mean()),above_half=int((values>=.5).sum()))
    save(out,dict(protocol='Ground-truth-assisted best region per eligible surface; candidates may be reused. Raw stage means projected native model labels at stride 1. No assignment is passed to prediction.',
                  source_manifest_sha256=digest(BASE/'manifest.json'),summary=summary,cases=records))
    print(summary,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--split',default='development')
    cli.add_argument('--tokens',type=int,nargs='+',default=[1600])
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    run(args.split,args.tokens,args.out)
