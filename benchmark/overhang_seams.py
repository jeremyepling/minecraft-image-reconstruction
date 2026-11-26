"""Measure raw segment evidence at shared underside boundaries and on flat surfaces."""
import numpy as np

from benchmark.camera_round import ROOT, read, save, digest
from benchmark.normal_round import groups
from benchmark.mask_round import load_lines
from benchmark.perimeter_sheet import load_parts
from benchmark.surface_detection_maps import detection_maps
from benchmark.surface_reference import recorded, ARCHIVE
from runtime.mode import image_only

RUN = ROOT/'benchmark/reviews/perimeters-01'
LINES = ROOT/'benchmark/data/mask-lines-v1'
NAMES = ('scalelsd','deeplsd')


def observable_seams(ids, pid):
    """Midpoint of every pixel interface with a *different building surface*, grouped by neighbor ID."""
    parts = {}
    for axis in (0, 1):
        a, b = (ids[:-1], ids[1:]) if axis==0 else (ids[:,:-1],ids[:,1:])
        for here, there in ((a,b),(b,a)):
            yy,xx = np.where((here==pid) & (there!=pid) & (there!=0))
            neighbor = there[yy, xx]
            for p in np.unique(neighbor):
                keep = neighbor == p
                xy = np.column_stack((xx[keep]+(.5 if axis==0 else 1.), yy[keep]+(1. if axis==0 else .5)))
                parts.setdefault(int(p), []).append(xy)
    return {p:np.concatenate(v) for p,v in parts.items()}


def tangents(xy):
    """Use local interface samples to estimate the tangent; remove tiny, ambiguous corners."""
    good, vectors = [], []
    for point in xy:
        nearby = xy[((xy-point)**2).sum(1)<=36]
        if len(nearby)<5:
            continue
        centered = nearby-nearby.mean(0)
        vals,vecs = np.linalg.eigh(centered.T@centered/len(centered))
        if vals[1] >= 3. and vals[1] >= 6 * max(vals[0],.001):
            good.append(point)
            vectors.append(vecs[:,-1])
    return np.asarray(good).reshape(-1,2),np.asarray(vectors).reshape(-1,2)


def support(xy, tang, lines):
    seg = np.asarray(lines,float).reshape(-1,2,2)
    vector = seg[:,1]-seg[:,0]
    length = np.linalg.norm(vector,axis=1)
    unit = vector/np.maximum(length[:,None],1.e-9)
    flags = np.zeros(len(xy),bool)
    for start in range(0,len(xy),256):
        points = xy[start:start+256]
        delta = points[:,None,:]-seg[None,:,0,:]
        along = np.clip((delta*unit).sum(2),0,length)
        dist2 = ((delta-along[:,:,None]*unit)**2).sum(2)
        parallel = np.abs(tang[start:start+256] @ unit.T) >= np.cos(np.deg2rad(15))
        flags[start:start+256] = np.any((dist2<=9.)&parallel,axis=1)
    return flags


def line_interior(ids,lines):
    """Count ~2px intervals of line inside building surfaces, including stable interior vs all."""
    edge = np.zeros_like(ids,dtype=bool)
    delta = ids[:-1]!=ids[1:]
    edge[:-1]|=delta; edge[1:]|=delta
    delta = ids[:,:-1]!=ids[:,1:]
    edge[:,:-1]|=delta; edge[:,1:]|=delta
    padded = np.pad(edge,5)
    near = np.zeros_like(edge)
    for dy in range(11):
        for dx in range(11):
            near |= padded[dy:dy+ids.shape[0],dx:dx+ids.shape[1]]
    covered = inner = 0
    for a,b in np.asarray(lines,float).reshape(-1,2,2):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/2.)))
        xy=a+((np.arange(n)+.5)/n)[:,None]*(b-a)
        xy = np.floor(xy).astype(int)
        xy = xy[((xy>=0)&(xy<ids.shape[::-1])).all(1)]
        on = ids[xy[:,1],xy[:,0]]>0
        covered += int(on.sum())
        inner += int((on&~near[xy[:,1],xy[:,0]]).sum())
    return [inner,covered]


@image_only()
def audit(output):
    manifest = read(RUN/'manifest.json')
    cohorts, _, fixtures = groups()
    assert fixtures == manifest['fixtures'] and digest(ARCHIVE)==manifest['reference_sha256']
    assert digest(RUN/'cases.json')==manifest['artifacts']['cases.json']
    cases = {c.floor+'/'+c.id:c for cc in cohorts.values() for c in cc}
    declarations={n:read(LINES/n/'manifest.json') for n in NAMES}
    rows=[]
    for row in read(RUN/'cases.json'):
        key = row['key']
        c=cases[key]
        _,ref = recorded(c)
        assert digest(RUN/row['predictions'])==manifest['artifacts'][row['predictions']]
        assert digest(c.path('image'))==row['sha256']
        parts = load_parts(RUN,row,None,None,'color-edges')
        ids,_,det = detection_maps(c,parts,row['scores']['color-edges'])
        lines = {n:load_lines(LINES,n,row['sha256'],ids.shape[::-1],declarations[n])['segments'] for n in NAMES}
        frame = dict(frame=key,layout=row['layout'],sha256=row['sha256'],undersides=[],
                     line_interior={n:line_interior(ids,raw) for n,raw in lines.items()})
        records = {r['id']:r for r in det['references']}
        for pid, patch in ref['patches'].items():
            if patch['normal']!=[0,-1,0]:
                continue
            item=dict(reference=pid,status=records[pid]['status'],pixels=records[pid]['pixels'],
                      reason=records[pid]['reason'],neighbors=[])
            for neighbor,xy in observable_seams(ids,pid).items():
                points,tang=tangents(xy)
                masks={n:support(points,tang,lines[n]) for n in NAMES}
                item['neighbors'].append(dict(id=neighbor,all_samples=len(xy),tested=len(points),
                    scalelsd=int(masks['scalelsd'].sum()),deeplsd=int(masks['deeplsd'].sum()),
                    either=int((masks['scalelsd']|masks['deeplsd']).sum())))
            frame['undersides'].append(item)
        rows.append(frame)
    save(output,dict(source_manifest_sha256=digest(RUN/'manifest.json'),
         fixtures=fixtures,reference_archive_sha256=digest(ARCHIVE),line_manifests={n:digest(LINES/n/'manifest.json') for n in NAMES},
         tolerance_px=3,parallel_tolerance_deg=15,tangent_radius_px=6,
         units='Native horizontal/vertical pixel interfaces; count shared seams with another visible building surface, excluding background. Local PCA retains sufficiently straight stretches. At each tested midpoint an aligned segment from either detector may vote.',
         negative_control='Approximately two-pixel samples along emitted source lines on the building. line_interior=[samples with no boundary pixel within a 5-pixel square radius, all on-building samples].',
         frames=rows))
