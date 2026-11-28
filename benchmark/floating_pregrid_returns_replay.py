"""Compare the two measured partitions through a fresh local atlas fit and ordinary assembly."""
import numpy as np
from PIL import Image
from unittest.mock import patch
from zipfile import ZipFile
from io import BytesIO
from hashlib import sha256

from benchmark import floating_pregrid_visible_returns as experiment
from benchmark import floating_package_rechecks as package

pink = experiment.pink


def refit(chart, labels, rgb, normal, whole):
    cells = chart['cells']
    uv = np.array([(c['u']+.5,c['v']+.5) for c in cells])
    centers = pink.prior.project(np.array(chart['H']),uv).astype(int)
    part_ids = labels[centers[:,1],centers[:,0]]
    vanishing = pink.prior.garden.pencils(rgb,pink.prior.garden.line_segments(rgb))
    result, checks = [], []
    for k in np.union1d(np.unique(labels[labels>0]),np.unique(part_ids)):
        rows = [c for c,assigned in zip(cells,part_ids) if assigned==k]
        sub = dict(chart,id=f"{chart['id']}~p{k}",cells=rows)
        candidates = []
        if k:
            region = labels==k
            measured = pink._unit(np.median(normal[region],axis=0))
            frames = []
            if measured @ chart['normal_camera'] > .92:
                frames.append(pink.prior.rectify(whole['quad'],whole['evidence'].get('aspect'),rgb,region))
            frames.append(pink.tangent_frame(rgb,region,measured,vanishing))
            for frame in frames:
                if frame is None:
                    continue
                found, ev = pink.prior.region_ruler(dict(sub,normal_camera=measured.tolist()),region,rgb,None,None,rectified=frame)
                if 'refit' in found:
                    holdout=ev['evidence']['refit']['holdout']
                    candidates.append(((len(holdout),np.mean([v['corr'] for v in holdout])),found['refit']))
        if candidates:
            result.append(max(candidates,key=lambda pair:pair[0])[1])
        elif rows:
            result.append(sub)
        checks.append(dict(part=int(k),old=len(rows),refitted=bool(candidates),emitted=len(result[-1]['cells']) if rows or candidates else 0))
    return result, checks


def replay(g, case):
    name = case['id']
    src = pink.data(g)[0]
    cid = experiment.TARGETS[name]
    whole = next(p for p in pink.read(src/'sky'/name/'source-handoff/counts.json')['surfaces'] if p['id']==cid)
    packet = pink.read(package.PACKAGE/g/'charts/sky'/name/'charts.json')
    chart = next(c for c in packet['charts'] if c['id']==cid)
    archived = pink.read(pink.WORKTREES/'floating-package-rechecks/runs/package-rechecks'/g/'cases'/(name+'.json.gz'))
    assert not archived['audit']['current_split']
    im = Image.open(case['variants']['sky']['image']).convert('RGB')
    rgb = np.asarray(im)
    normal,_,_ = pink.source_fields(g,case,src,im.size)
    folder = experiment.OUT/g/'source'/name
    roots,tests = {},{}
    source_sky = package.sky_support(im)
    for arm in ('noop','baseline','returns'):
        charts = packet['charts']
        if arm!='noop':
            labels=np.asarray(Image.open(folder/(cid+'-'+arm+'.png')))
            pieces,tests[arm] = refit(chart,labels,rgb,normal,whole)
            charts = [part for c in charts for part in (pieces if c['id']==cid else [c])]
        charts,_ = package.border_filter(charts,source_sky)
        root=experiment.OUT/g/arm
        pink.save(root/'charts/sky'/name/'charts.json',dict(packet,charts=charts))
        with patch.object(pink.combined,'inputs',pink.data):
            pink.combined.run((g,case,root/'charts',root/'assembly'))
        roots[arm]=root/'assembly'
    records = {arm:pink.read(path/pink.PREFIX/name/'score.json') for arm,path in roots.items()}
    archive = pink.WORKTREES/'floating-visible-builds/benchmark/reviews/floating-expanded-benchmark-01'
    manifest = pink.read(archive/'manifest.json')
    target = next(r for r in manifest['conditions'] if (r['panel'],r['id'])==(g,name))
    assert pink.digest(archive/'exports.zip') == manifest['archive']['sha256']
    with ZipFile(archive/'exports.zip') as outer:
        original = outer.read(target['files']['pack']['entry'])
    assert sha256(original).hexdigest() == archived['audit']['pack_sha256']['split-border'] == target['files']['pack']['sha256']
    with ZipFile(BytesIO(original)) as frozen, ZipFile(roots['noop']/pink.PREFIX/name/'reconstruction.zip') as replayed:
        assert set(frozen.namelist())==set(replayed.namelist())
        assert all(frozen.read(member)==replayed.read(member) for member in frozen.namelist())
    assert all(v['reference_sha256']==records['noop']['reference_sha256'] for v in records.values())
    scored = pink.summarize([case],roots,pink.PREFIX)['rows'][0]
    aligned = {k:package.read_arm(r/pink.PREFIX/name)[1] for k,r in roots.items()}
    pink.save(experiment.REVIEW/(name+'-placements.json'),dict(scored,regional_fit=tests,
              packs={k:v['pack_sha256'] for k,v in records.items()},reference=records['noop']['reference_sha256'],
              changed_sites=package.sites(case,aligned['baseline'],aligned['returns'])))
