"""Compare single-image geometry to the active default on admitted whole-structure views."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

from PIL import Image

from benchmark import canopy_fixture_metrics as openings, relative_canopy_spacing as spacing
from benchmark import visible_depth_native as native, visible_depth_transfer as transfer, visible_depth_score
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.nearby_thin_comparison import EVALUATION, EXTRA_INPUTS, canopy_report
from benchmark.relative_geometry import placement_scores
from benchmark.relative_geometry_replay import references
from benchmark.texture_camera_eval import ROOT, STABLE, TRAINING_DATA, TRAINING_RELEASE, checked, digest, frames, read, save
from benchmark.visible_recovery_report import summarize


REVIEW = ROOT / 'benchmark/reviews'
ARCHIVE = REVIEW / 'nearby-thin-01/scores.zip'
NEW_ARCHIVE = REVIEW / 'blind-shapes-01/scores.zip'
NEW_INPUTS = EXTRA_INPUTS.parents[2] / 'visible-geometry-new-shapes/runs/blind-input'
DEPTH = STABLE / 'runs/visible-depth-rounds'
SIZES = dict(train=72, validation=12, test=12, regression=48, native=12, old=12, fresh=8, extra=20, new=4)
NATIVE_INPUTS = dict(old=DEPTH/'transfer', fresh=STABLE/'runs/canopy-placement/confirmation',
                     extra=EXTRA_INPUTS, new=NEW_INPUTS)


@lru_cache(maxsize=2)
def archive(path):
    return ZipFile(path)


def members(split):
    return (archive(NEW_ARCHIVE), '2strip') if split=='new' else (archive(ARCHIVE), split)


def cases(splits):
    """Enumerate independently approved input views and their archived baseline rows."""
    external = read(REVIEW/'whole-structure-framing-01/evaluation-only.json')['cohorts']
    for split in splits:
        z, folder = members(split)
        manifest = json.loads(z.read(f'{folder}/manifest.json'))
        encoded = z.read(f'{folder}/cases.json')
        assert sha256(encoded).hexdigest() == manifest['cases_sha256']
        rows = {r['id']:r for r in json.loads(encoded)}
        if split in NATIVE_INPUTS:
            screen = NATIVE_INPUTS[split]/'screen.json'
            assert digest(screen) == manifest['selection_sha256']
            sample = read(screen)['selected']
        elif split=='native':
            sample = native.rows()
        else:
            sample = frames(TRAINING_DATA,split,TRAINING_RELEASE)
        ids = {f['id']:f['files']['rgb.png'] for f in sample}
        assert len(sample)==len(ids)==SIZES[split]
        if split in external:
            approved = {r['id']:r['rgb_sha256'] for r in external[split] if r['admitted']}
            assert ids==approved
        for f in sample:
            baseline = rows[f['id']]
            assert baseline['rgb_sha256']==ids[f['id']]
            yield split,f,baseline


def photo(split, f, prior):
    if split in NATIVE_INPUTS:
        root=NATIVE_INPUTS[split]
        inputs=root/'upstream'/f['id']
        parts = read(checked(inputs/'surfaces.json',prior['surface_sha256']))['planes']
        counts = read(checked(inputs/'counts.json',prior['counts_sha256']))
        camera = f['camera']
        captures = NEW_INPUTS if split=='new' else transfer.BUILDS
        rgb = captures/f['directory']/'captures'/(f['source_id']+'.png')
    else:
        measured=read(checked(DEPTH/'inputs'/(f['id']+'.json'),prior['input_sha256']))
        assert measured['id']==f['id'] and measured['rgb_sha256']==prior['rgb_sha256']
        parts,counts,camera = (measured[k] for k in ('parts','counts','camera'))
        rgb = native.rgb_path(f,native.BUILDS) if split=='native' else TRAINING_DATA/'data'/split/f['directory']/'rgb.png'
    with Image.open(checked(rgb,prior['rgb_sha256'])) as src:
        return src.convert('RGB'),parts,counts,camera


@lru_cache(maxsize=1)
def roof_specs():
    old = read(REVIEW/'visible-recovery-01/spacing/cases.json')['cases']
    alt = read(REVIEW/'visible-recovery-extra-views-01/roofs.json')
    specs = read(openings.PROBES)
    out = {r['id']:specs[r['id']] for r in old}
    out.update({r['id']:specs[r['previous_probe']] for r in alt})
    return out


def roof(f, cells, target):
    spec=roof_specs()[f['id']]
    rays=openings.score(f,cells,spec)['rays']
    return dict(spacing=spacing.score(cells,target,spec),clear_rays=len(rays),
                filled_clear_rays=sum(r['foreground_filled'] for r in rays))


@lru_cache(maxsize=12)
def prediction_records(directory, manifest):
    meta = read(directory/manifest)
    rows = read(checked(directory/'cases.json',meta['cases_sha256']))
    return {(r.get('split'),r['id']):r for r in rows}


def prior_candidate(root,split,key,prior):
    if (root/'summary.json').is_file():
        stored=prediction_records(root,'summary.json')[split,key]
    else:
        rows=prediction_records(root/split,'manifest.json')
        stored=rows[(split,key) if (split,key) in rows else (None,key)]
    assert stored['rgb_sha256']==prior['rgb_sha256'],key
    hashes=stored.get('input_hashes',stored)
    assert all(hashes[k]==prior[k] for k in ('input_sha256','surface_sha256','counts_sha256') if k in prior),key
    return checked(root/split/key/'reconstruction.zip',stored['pack_sha256'])


def score(job):
    split,f,prior,out,predictions = job
    key=f['id']
    z,folder=members(split)
    stem=f'{folder}/{key}'
    pack=z.read(stem+'/reconstruction.zip')
    assert sha256(pack).hexdigest()==prior['pack_sha256']
    before_local=set(decoded_pack(BytesIO(pack)))
    stored=json.loads(z.read(stem+'/cells.json'))
    assert before_local==set(map(tuple,stored['local']))
    dest=out/split/key
    dest.mkdir(parents=True,exist_ok=True)
    pack_path=dest/'reconstruction.zip'
    status={}
    if predictions is not None:
        proposed=prior_candidate(predictions,split,key,prior)
        pack=proposed.read_bytes()
        local=set(decoded_pack(BytesIO(pack)))
        pack_path.write_bytes(pack)
    elif visible_depth_score.predict is None:
        local=before_local
        pack_path.write_bytes(pack)
    else:
        image,parts,counts,camera=photo(split,f,prior)
        answer=visible_depth_score.predict(image,parts,counts,camera,before_local.copy())
        cells,status=answer if isinstance(answer,tuple) else (answer,{})
        local=set(cells)
        if local==before_local:
            pack_path.write_bytes(pack)
        else:
            local=_write_read(pack_path,local)

    # Open world and recorded camera only after predictions are exported.
    ref_split='old' if split in ('extra','new') else split
    builds=NEW_INPUTS if split=='new' else transfer.BUILDS
    truth,occupied,transform,hashes=references(ref_split,f,NATIVE_INPUTS.get(split),TRAINING_DATA,DEPTH,
        dict(native=native.BUILDS,old=builds,fresh=builds))
    assert all(prior[k]==v for k,v in hashes.items()) and (transform is not None or not local)
    before,after=registered(before_local,transform),registered(local,transform)
    assert before==set(map(tuple,stored['world']))
    counts=placement_scores(after,before,dict.fromkeys(truth),occupied)
    for field in ('metrics','relative_geometry'):
        m=counts[field]
        del m['joint_correct']
        assert m['target']==prior[field]['target']
        assert prior[field]['correct']==m['correct']-m['gained']+m['lost']
        assert prior[field]['extras']==m['extras']-m['extra_added']+m['extra_removed']
        assert prior[field]['hidden_extras']==m['hidden_extras']-m['hidden_added']+m['hidden_removed']
    assert list(counts['baseline_translation'])==prior['translation']
    def span(points):
        return [max(p[i] for p in points)-min(p[i] for p in points)+1 for i in range(3)] if points else [0]*3
    x_min,x_max=min(p[0] for p in occupied),max(p[0] for p in occupied)
    z_min,z_max=min(p[2] for p in occupied),max(p[2] for p in occupied)
    def outside_footprint(points,shift):
        return sum(not (x_min<=x+shift[0]<=x_max and z_min<=z+shift[2]<=z_max) for x,y,z in points)
    old=prior['relative_geometry']
    m=counts['relative_geometry']
    row=dict(split=split,id=key,family=f.get('family',f.get('layout',f.get('shape'))),
             rgb_sha256=prior['rgb_sha256'],input_hashes=hashes,reference_hashes=f['files'],
             control_pack_sha256=prior['pack_sha256'],pack_sha256=digest(pack_path),changed=before_local!=local,
             previous_f1=2*old['correct']/(old['target']+old['placed']),f1=2*m['correct']/(m['target']+m['placed']),
             span=dict(target=span(truth),before=span(before),after=span(after)),
             outside_source_xz=dict(before=outside_footprint(before,counts['baseline_translation']),
                                    after=outside_footprint(after,counts['translation'])),status=status,**counts)
    if key in roof_specs():
        row['previous_roof']=roof(f,before,truth)
        row['roof']=row['previous_roof'] if before==after else roof(f,after,truth)
    save(dest/'cells.json',dict(local=sorted(local),world=sorted(after)))
    return row


def compare(args):
    splits=args.splits
    listed=list(cases(splits))
    if args.ids:
        listed=[t for t in listed if t[1]['id'] in args.ids]
        if {t[1]['id'] for t in listed}!=set(args.ids):
            raise ValueError('Requested frame IDs missing from selected splits')
    if args.predictions and args.predictions.resolve()==args.out.resolve():
        raise ValueError('Use a different directory for rescoring')
    jobs=[(*t,args.out,args.predictions) for t in listed]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=visible_depth_score.init_candidate,
                             initargs=(args.candidate_root.resolve(),args.candidate)) as pool:
        rows=list(pool.map(score,jobs))
    groups={s:[r for r in rows if r['split']==s] for s in splits}
    if set(EVALUATION)<=set(splits):
        groups['fixed']=[r for r in rows if r['split'] in EVALUATION]
    groups={k:v for k,v in groups.items() if v}
    aggregates={k:summarize(rr) for k,rr in groups.items()}
    for k,rr in groups.items():
        aggregates[k]['changed']=sum(r['changed'] for r in rr)
        aggregates[k]['outside_source_xz']={name:sum(r['outside_source_xz'][name] for r in rr)
                                          for name in ('before','after')}
    results=dict(control='acfe46b',default_engine='36d406a',candidate=args.candidate,
                 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.candidate_root,text=True).strip(),
                 predictions=str(args.predictions) if args.predictions else None,
                 primary='Pooled visible-block F1 after one common XYZ translation',
                 sources={p.name if p==TRAINING_RELEASE else p.parent.name:digest(p) for p in
                          (TRAINING_RELEASE,ARCHIVE,NEW_ARCHIVE,REVIEW/'whole-structure-framing-01/evaluation-only.json')},
                 groups=aggregates,canopies={k:{field:canopy_report([r[field] for r in rr if field in r])
                              for field in ('previous_roof','roof')} for k,rr in groups.items() if any('roof' in r for r in rr)},
                 worst={k:[r['id'] for r in sorted(rr,key=lambda r:(r['f1'],r['id']))[:3]] for k,rr in groups.items()})
    if args.candidate:
        src=args.candidate_root.joinpath(*args.candidate.split(':')[0].split('.')).with_suffix('.py')
        results['adapter_sha256']=digest(src)
        results['candidate_worktree_status']=subprocess.check_output(['git','status','--porcelain'],cwd=args.candidate_root,text=True).splitlines()
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'cases.json',rows)
    results['cases_sha256']=digest(args.out/'cases.json')
    save(args.out/'summary.json',results)
    for k in ('train','fixed','extra','new'):
        if k in aggregates:
            m=aggregates[k]
            print(k,m['frames'],f"{m['f1']:.4%}",'gained/lost',m['gained'],m['lost'],'extras',m['extras'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',choices=SIZES,default=list(SIZES))
    p.add_argument('--ids',nargs='+',help='Run only the listed frame IDs from those splits')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--candidate-root',type=Path,default=ROOT)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--candidate',help='module:function(image, parts, counts, camera, prior)')
    source.add_argument('--predictions',type=Path,help='Root of completed split/id/reconstruction.zip exports')
    compare(p.parse_args())
