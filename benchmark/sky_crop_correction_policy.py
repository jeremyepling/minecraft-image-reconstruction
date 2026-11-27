"""Learn successive source-image edits for native visible-surface crop masks."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.sky_crop_chooser import DEV, MODEL_RUN
from benchmark.sky_crop_options import ACTIVE, SKY, WORKTREES, fields, native
from engine.floating_sky import sky_support
from engine.plane_consensus import row_runs


OUT = ROOT/'runs/sky-crop-correction-policy'
CHOOSER = ROOT/'benchmark/reviews/sky-crop-chooser-01/learned.json.gz'
REFERENCE = WORKTREES/'sky-crop-reference/benchmark'
STEPS, RADIUS, SEED = 4, 5, 4101
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*RADIUS+1,2*RADIUS+1))
FEATURES = ['current','current_fraction_7','current_fraction_21','original',
            'is_sky','sam_box','sam_points','sam_zoom','mobile_points',
            'inside_rgb_error','outside_rgb_error','inside_minus_outside_rgb_error',
            'inside_local_depth_error','outside_local_depth_error','core_normal_agreement',
            'core_plane_depth_error','local_rgb_gradient']
SAMS = ('sam-b-box','sam-b-points','sam-b-zoom','mobile-points')


@dataclass
class Crop:
    case: str
    ident: str
    size: tuple
    roi: tuple
    original: np.ndarray
    initial: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    sky: np.ndarray
    sam: list
    gradient: np.ndarray
    target: np.ndarray | None = None
    target_area: int = 0


def surface_targets(case, parts, size):
    """Open references only for the two training buildings; lock targets to ordinary supports."""
    if case not in DEV:
        raise ValueError('Training may access only development labels')
    import benchmark
    if str(REFERENCE) not in benchmark.__path__:
        benchmark.__path__.append(str(REFERENCE))
    from benchmark.sky_crop_reference import best_target, intersection, reference_raster
    descriptor = next(c for c in read(SKY/'fixtures.json')['cases'] if c['id']==case)
    frame = Path(descriptor['frame'])
    if any(digest(frame/n)!=v for n,v in descriptor['reference_sha256'].items()):
        raise ValueError(f'Changed development reference {case}')
    ref = reference_raster(frame)
    paired = {}
    for part in parts:
        mask = native(part['support'],size)
        hits = intersection(ref,mask)
        index = best_target(ref.areas,hits,int(mask.sum()))
        if index:
            paired[part['id']] = (index,int(ref.areas[index]))
    return ref.labels, paired


def input_crops(case, packets, chooser, start, training):
    folder = next((SKY/'baseline/sky').glob(f'*/{case}'))
    image_file = SKY/'images'/case/'sky.png'
    handoff = ACTIVE/case/'source-handoff'
    image_hash = digest(image_file)
    if image_hash != read(handoff/'build-manifest.json')['image_sha256'] or image_hash != read(folder/'receipt.json')['source_sha256']:
        raise ValueError(f'Wrong image for {case}')
    image = Image.open(image_file).convert('RGB')
    rgb = np.asarray(image, np.uint8)
    norm, z, _ = fields(folder,image.size)
    sky = sky_support(image)
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0), cv2.Sobel(gray,cv2.CV_32F,0,1))/1020.
    parts = read(handoff/'surfaces.json')['planes']
    labels, paired = surface_targets(case,parts,image.size) if training else (None,{})
    crops = []
    for part in parts:
        ident = part['id']
        if training and ident not in paired:
            continue
        original = native(part['support'],image.size)
        support = chooser.get(case,{}).get(ident) if start=='chooser' else None
        initial = original if support is None else native(support,image.size)
        # Retain the full starting support and reachable neighbourhood of every future boundary.
        y,x = np.nonzero(initial|original)
        pad = STEPS*RADIUS + 13
        l,r = max(0,int(x.min())-pad), min(image.width,int(x.max())+pad+1)
        t,b = max(0,int(y.min())-pad), min(image.height,int(y.max())+pad+1)
        sl = np.s_[t:b,l:r]
        options = [native(packet[case][ident],image.size)[sl].copy() for packet in packets]
        crop = Crop(case,ident,image.size,(l,t,r,b),original[sl].copy(),initial[sl].copy(),rgb[sl],
                    z[sl],norm[sl],sky[sl],options,grad[sl])
        if training:
            index, crop.target_area = paired[ident]
            crop.target = labels[sl] == index
        crops.append(crop)
    return crops


def average(a):
    return cv2.boxFilter(a,cv2.CV_32F,(21,21),normalize=True,borderType=cv2.BORDER_REPLICATE)


def samples(crop, mask, rng=None):
    m = mask.astype(np.uint8)
    band = (cv2.dilate(m,KER) != cv2.erode(m,KER))
    y,x = np.nonzero(band)
    if rng is not None and len(x)>1200:
        ix = rng.choice(len(x),size=1200,replace=False)
        y,x = y[ix],x[ix]
    if not len(x):
        return np.empty((0,len(FEATURES)),np.float32),y,x
    mf = m.astype(np.float32)
    fraction = average(mf)
    neighborhood = cv2.boxFilter(mf,-1,(7,7))
    rgb = crop.rgb.astype(np.float32) / 255.
    # The current mask changes the interior/exterior texture and depth prototypes on every pass.
    def two_errors(field):
        mass = average(field * (mf[:,:,None] if field.ndim==3 else mf))[y,x]
        total = average(field)[y,x]
        n = fraction[y,x]
        if field.ndim==3:
            n = n[:,None]
        sample = field[y,x]
        inside = np.abs(sample - mass / np.maximum(n,1e-5))
        outside = np.abs(sample - (total-mass)/np.maximum(1-n,1e-5))
        if field.ndim==3:
            inside,outside = inside.mean(axis=1),outside.mean(axis=1)
        return inside,outside
    rgb_in,rgb_out = two_errors(rgb)
    z_in,z_out = two_errors(crop.depth)
    iy,ix = np.nonzero(mask)
    take = np.linspace(0,len(ix)-1,min(512,len(ix)),dtype=int)
    iy,ix = iy[take],ix[take]
    n = np.median(crop.normal[iy,ix],axis=0)
    n /= max(1.e-6,np.linalg.norm(n))
    design = np.column_stack((ix,iy,np.ones(len(ix))))
    coef = np.linalg.lstsq(design,crop.depth[iy,ix],rcond=None)[0]
    pred = coef[0]*x+coef[1]*y+coef[2]
    values = np.column_stack((mask[y,x],neighborhood[y,x],fraction[y,x],crop.original[y,x],
                              crop.sky[y,x],*(s[y,x] for s in crop.sam),rgb_in,rgb_out,rgb_in-rgb_out,
                              z_in,z_out,crop.normal[y,x]@n,abs(crop.depth[y,x]-pred),crop.gradient[y,x]))
    return values.astype(np.float32),y,x


def rows(crops, masks):
    rng = np.random.default_rng(SEED)
    data,targets = [],[]
    for crop,m in zip(crops,masks):
        features,y,x = samples(crop,m,rng)
        if len(x):
            data.append(features)
            targets.append(crop.target[y,x].astype(np.int32))
    return np.vstack(data),np.concatenate(targets)


def fit(inputs):
    x = np.concatenate([v[0] for v in inputs])
    y = np.concatenate([v[1] for v in inputs])
    cv2.setRNGSeed(SEED)
    forest = cv2.ml.RTrees_create()
    forest.setMaxDepth(7)
    forest.setMinSampleCount(70)
    forest.setActiveVarCount(6)
    forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,32,0))
    forest.setCalculateVarImportance(True)
    if not forest.train(x,cv2.ml.ROW_SAMPLE,y):
        raise ValueError('Correction policy training failed')
    return forest


def advance(model, crop, current):
    features,y,x = samples(crop,current)
    result = current.copy()
    if len(x):
        votes = model.getVotes(features,0)
        positive = votes[1:,votes[0]==1].ravel()/votes[1:].sum(axis=1)
        result[y[positive>=.70],x[positive>=.70]] = True
        result[y[positive<=.30],x[positive<=.30]] = False
    return result


def trajectory(model,crops):
    states = [[c.initial for c in crops]]
    for _ in range(STEPS):
        states.append([advance(model,c,m) for c,m in zip(crops,states[-1])])
    return states


def teach(crops):
    """Fit initial mistakes, then train on residual mistakes produced by two real student passes."""
    initial = [c.initial for c in crops]
    observations = [rows(crops,initial)]
    first = fit(observations)
    current = initial
    for _ in range(2):
        current = [advance(first,c,m) for c,m in zip(crops,current)]
        observations.append(rows(crops,current))
    return fit(observations)


def train(out, packets):
    crops = {c:input_crops(c,packets,{},'ordinary',True) for c in DEV}
    cross_scores = {}
    for train_id,test_id in (DEV,DEV[::-1]):
        model = teach(crops[train_id])
        held = crops[test_id]
        track = trajectory(model,held)
        cross_scores[test_id] = [float(np.mean([overlap(c,m) for c,m in zip(held,state)])) for state in track]
    development_curve = np.mean(list(cross_scores.values()),axis=0)
    # Earliest step wins ties. The decision is frozen before either held-out inventory is inferred.
    step = int(np.argmax(development_curve))
    model = teach(crops[DEV[0]]+crops[DEV[1]])
    out.mkdir(parents=True,exist_ok=True)
    model.save(str(out/'forest.xml.gz'))
    save(out/'freeze.json',dict(dev_ids=DEV,seed=SEED,radius_px=RADIUS,steps=STEPS,
                                train_rollout_steps=2,trees=32,max_depth=7,
                                transfer_curves=cross_scores,mean_transfer_curve=development_curve.tolist(),
                                selected_round=step,add_probability=.70,remove_probability=.30,
                                model_sha256=digest(out/'forest.xml.gz'),
                                features=FEATURES,importance=model.getVarImportance().ravel().tolist(),
                                option_sha256={n:digest(MODEL_RUN/f'{n}.json') for n in SAMS}))
    print('Frozen development curves',cross_scores,'selected',step,flush=True)


def overlap(crop,mask):
    hits = np.count_nonzero(crop.target & mask)
    return hits/max(1,crop.target_area+mask.sum()-hits)


def pack(crop,mask):
    l,t,_,_ = crop.roi
    return dict(size=list(crop.size),runs=[[y+t,a+l,b+l] for y,a,b in row_runs(mask)])


def infer(out, packets, start):
    frozen = read(out/'freeze.json')
    if frozen['model_sha256'] != digest(out/'forest.xml.gz') or any(
            digest(MODEL_RUN/f'{n}.json')!=h for n,h in frozen['option_sha256'].items()):
        raise ValueError('Changed frozen model or segmentation packets')
    model = cv2.ml.RTrees_load(str(out/'forest.xml.gz'))
    chosen = read(CHOOSER)['frames'] if start=='chooser' else {}
    target_out = out if start=='ordinary' else out/'from-chooser'
    packets_by_round = [{} for _ in range(STEPS+1)]
    trace = {}
    for folder in sorted(ACTIVE.iterdir()):
        case = folder.name
        crops = input_crops(case,packets,chosen,start,False)
        states = trajectory(model,crops)
        for i,state in enumerate(states):
            packets_by_round[i][case] = {c.ident:pack(c,m) for c,m in zip(crops,state)
                                         if not np.array_equal(m,c.original)}
        frame = trace[case] = {}
        for k,c in enumerate(crops):
            history = frame[c.ident] = []
            for step in range(1,STEPS+1):
                before,after = states[step-1][k],states[step][k]
                history.append(dict(step=step,area_before=int(before.sum()),area_after=int(after.sum()),
                                    added=int((after & ~before).sum()),removed=int((before & ~after).sum())))
        print('Inferred',start,case,flush=True)
    for i,packet in enumerate(packets_by_round):
        save(target_out/f'round-{i}.json',dict(frames=packet))
    save(target_out/'selected.json',dict(frames=packets_by_round[frozen['selected_round']]))
    save(target_out/'trace.json',dict(frames=trace,initialization=start,selected_round=frozen['selected_round'],
                                      radius_px=RADIUS,model_sha256=frozen['model_sha256']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('train','infer'))
    parser.add_argument('--start',choices=('ordinary','chooser'),default='ordinary')
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    packets = [read(MODEL_RUN/f'{n}.json')['frames'] for n in SAMS]
    if args.action=='train':
        train(args.output, packets)
    else:
        infer(args.output, packets, args.start)


if __name__=='__main__':
    main()
