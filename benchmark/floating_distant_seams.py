"""Challenge a frozen source-face ruler using distant seam contrast in the same connected mask."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_one_face_control import (
    ROOT, WORKTREES, PANELS, base, inputs, combined, project, read, save, digest, summarize, _native_alpha)
from engine.geometry import bilinear

OUT = ROOT / 'runs/distant-seams'
REVIEW = ROOT / 'benchmark/reviews/floating-distant-seams-01'
GROUPS = (*PANELS, 'holdout3', 'holdout4')


def locations(group):
    if group == 'holdout5':
        return WORKTREES/'floating-extra-audit/runs/face-proposals/holdout5'
    key = {'holdout3':'new6','holdout4':'new7'}.get(group,group)
    return WORKTREES/'floating-face-proposals/runs/face-proposals'/key


def data(group):
    if group != 'holdout5':
        return inputs(group)
    loc = WORKTREES/'floating-extra-audit/runs/floating-holdout5'
    return loc/'ordinary',loc/'ordinary/atlas/connected',loc/'source_material/charts',read(loc/'manifest.json')['cases']


def common_domain(mask, original, candidate):
    # Stay inside a single connected source component containing the offered face.
    _, labels = cv2.connectedComponents(mask.astype('B'), connectivity=4)
    center = project(np.asarray(candidate['H']),np.array([[.5,.5]]))[0].astype(int)
    code = labels[center[1],center[0]]
    support = labels == code if code else np.zeros_like(mask)
    y,x = np.nonzero(support)
    xy = np.column_stack((x+.5,y+.5))
    # Exclude the proposed face and its adjacent verification tiles, as well as the old
    # origin tile and its neighbors. Both rulers are judged in the same source region.
    distant = np.ones(len(x),bool)
    for h in (original['H'],candidate['H']):
        uv = project(np.linalg.inv(h),xy)
        distant &= np.max(abs(uv-.5),axis=1) >= 2.5
    region = np.zeros_like(mask)
    region[y[distant],x[distant]] = True
    return cv2.erode(region.astype('B'),np.ones((3,3),'B')).astype(bool)


def measure(chart, rgb, domain):
    h = np.asarray(chart['H'])
    uv = np.asarray([(p['u'],p['v']) for p in chart['cells']],float).reshape(-1,2)
    out = []
    for axis in (0,1):
        # Each unit-length seam has two adjacent cells. Probe contrast across the seam
        # and across four displaced interior lines at the same local sampling scale.
        unit = np.eye(2)[axis]
        starts = np.unique(np.vstack((uv,uv+unit)),axis=0)
        sites = starts[:,None,:] + np.linspace(.14,.86,7)[None,:,None]*(1-unit)
        offset = np.array([0.,-.31,-.19,.19,.31])
        anchors = sites[:,:,None,:] + offset[None,None,:,None]*unit
        a = project(h,anchors)
        b = project(h,anchors+.02*unit)
        step = (b-a)/.02
        pitch = np.linalg.norm(step,axis=-1)
        # Cap comparison footprint at 1.25 px on either side, floor at half a pixel.
        disp = step*(np.clip(.065*pitch,.5,1.25)/np.maximum(pitch,1e-7))[...,None]
        ok = (pitch>=5)&np.isfinite(pitch)
        colors = []
        for xy in (a-disp,a+disp):
            ix,iy = np.moveaxis(np.floor(np.nan_to_num(xy)).astype(int),-1,0)
            inside = (ix>=0)&(ix<domain.shape[1])&(iy>=0)&(iy<domain.shape[0])
            ok &= inside & domain[iy.clip(0,domain.shape[0]-1),ix.clip(0,domain.shape[1]-1)]
            colors.append(bilinear(rgb,np.nan_to_num(xy)))
        contrast = np.minimum(48, np.mean(np.abs(colors[1]-colors[0]),axis=-1))
        good = ok.all(axis=-1)
        # Material changes may provide valid boundaries. Bound strong color jumps so
        # a single change of material cannot dominate repeated weaker block seams.
        effect = (contrast[:,:,0]-contrast[:,:,1:].mean(axis=2))/(6+contrast.mean(axis=2))
        count = good.sum(axis=1)
        valid = count>=4
        effects = (effect*good).sum(axis=1)[valid]/count[valid]
        out.append(dict(segments=len(effects),score=float(np.clip(effects,-1,1).mean()) if len(effects) else 0.))
    segments = sum(a['segments'] for a in out)
    score = sum(a['score']*a['segments'] for a in out)/max(1,segments)
    return dict(segments=segments,score=score,axes=out)


def run(job):
    group,case = job
    cv2.setNumThreads(1)
    src,_,whole,_ = data(group)
    name = case['id']
    rel = Path('sky')/name/'charts.json'
    loc = locations(group)
    raw = read(whole/rel)
    one = read(loc/'charts'/rel)
    offers = read(loc/'offers'/(name+'.json.gz'))
    assert one['source_packet_sha256']==digest(whole/rel)
    image = Path(case['variants']['sky']['image'])
    assert offers['source_sha256']==digest(image)
    rgb = np.asarray(Image.open(image).convert('RGB'))
    plane_file = src/'sky'/name/'source-handoff/surfaces.json'
    planes = {p['id']:p for p in read(plane_file)['planes']}
    picks = {r['id']:r['picked'] for r in offers['charts']}
    result,audit = [],[]
    for old,new in zip(raw['charts'],one['charts'],strict=True):
        cid = old['id']
        assert new['id']==cid
        winner = old
        if picks.get(cid) is not None:
            alpha,(x,y,x1,y1) = _native_alpha(planes[cid]['support'],rgb.shape[1::-1])
            mask = np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1] = np.asarray(alpha)>0
            domain = common_domain(mask,old,new)
            previous = measure(old,rgb,domain)
            proposed = measure(new,rgb,domain)
            # Require several independent far seams, positive absolute evidence and
            # a margin over the original ruler. Insufficient evidence retains it.
            use_new = (min(previous['segments'],proposed['segments'])>=4
                       and proposed['score']>.08 and proposed['score']>previous['score']+.05)
            if use_new:
                winner = new
            audit.append(dict(id=cid,old=previous,new=proposed,accepted=use_new,domain=int(domain.sum())))
        result.append(winner)
    dest = OUT/group
    save(dest/'charts'/rel,dict(raw,charts=result,proposals_sha256=digest(loc/'charts'/rel)))
    save(dest/'audit'/(name+'.json'),dict(id=name,source_sha256=digest(image),surface_sha256=digest(plane_file),
        original_sha256=digest(whole/rel),proposals_sha256=digest(loc/'charts'/rel),charts=audit))
    with patch.object(combined,'inputs',data):
        combined.run((group,case,dest/'charts',dest/'assembly'))
    return group,name,sum(r['accepted'] for r in audit)


def report(groups):
    for group in groups:
        baseline = base(group) if group!='holdout5' else data(group)[0].parent/'horizontal-cells/assembly'
        roots = dict(horizontal=baseline,source_one=locations(group)/'assembly',distant=OUT/group/'assembly')
        results = summarize(data(group)[3],roots,'connected-atlas/sky')
        save(REVIEW/(group+'.json.gz'),results)
        audits = [read(OUT/group/'audit'/(case['id']+'.json')) for case in data(group)[3]]
        save(REVIEW/(group+'-audits.json.gz'),audits)
        print(group,{k:round(v['geometry_strict']['f1']*100,2) for k,v in results['cohorts']['all'].items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('groups',nargs='*',default=GROUPS)
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--resume',action='store_true')
    args = cli.parse_args()
    jobs = [(g,c) for g in args.groups for c in data(g)[3]]
    if args.resume:
        jobs = [(g,c) for g,c in jobs if not (OUT/g/'assembly/connected-atlas/sky'/c['id']/'score.json').exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(run,jobs):
            print(result,flush=True)
    report(args.groups)
