"""Replay local inset edits on material-bearing, decoded packs and grade visible coordinates."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import benchmark
from benchmark import framed_depth_eval as framed
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_priority_comparison import compare as compare_priority
from benchmark.texture_camera_eval import checked, digest, read, save
from engine.export import JAVA_26_3_FORMAT, write_pack
from engine.visible_inset_cells import inset_edits

HERE=Path(__file__).resolve().parents[1]
if not (HERE/'benchmark/facade_inset_round.py').exists():
    benchmark.__path__.append(str(HERE.parent/'facade-inset-round/benchmark'))
from benchmark import facade_inset_round as suite

CANDIDATE=HERE.parent/'facade-inset-round/runs/returns'
METHODS=('add_only','clear_front')
FIELDS=('correct','missed','extras','hidden_extras','gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')


def case(job):
    split,frame,prior,control,event,out=job
    key=frame['id']
    assert event['rgb_sha256']==control['rgb_sha256']==frame['files']['rgb.png']
    if split=='shelter':
        with ZipFile(suite.ARCHIVE) as archive:
            packed=BytesIO(archive.read(f'production/transfer/{key}/reconstruction.zip'))
        import hashlib
        assert hashlib.sha256(packed.getvalue()).hexdigest()==control['pack_sha256']
        root,inputs,ref=suite.SHELTER,suite.SHELTER,'old'
    else:
        packed=checked(suite.CONTROL/split/key/'reconstruction.zip',control['pack_sha256'])
        root=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
        inputs=framed.NATIVE_INPUTS.get(split)
        ref='old' if split in ('extra','new') else split
    original=decoded_pack(packed)
    assert set(original.values())<={'minecraft:smooth_stone'}
    base={k:control['relative_geometry'][k] for k in FIELDS[:4]}
    record=dict(id=key,split=split,rgb_sha256=event['rgb_sha256'],control_pack_sha256=control['pack_sha256'],baseline=base,methods={})
    if not event['changed']:
        return record
    # Current surfaces are read from the checksum-verified detector output below.
    # The ordinary facade and watchpost repairs preserved the frozen image and estimated camera.
    image,_,_,camera=suite.ordinary_inputs(split,frame,prior,control) if split=="shelter" else framed.photo(split,frame,prior)
    path=checked(CANDIDATE/split/key/'surfaces.json',event['surfaces_sha256'])
    record['surfaces_sha256']=digest(path)
    parts=read(path)['planes']
    added,removed=inset_edits(image.size,camera,parts,event['events'],original)
    truth,occupied,transform,_=framed.references(ref,frame,inputs,framed.TRAINING_DATA,framed.DEPTH,
        dict(native=framed.native.BUILDS,old=root,fresh=root))
    before=registered(original,transform)
    for name in METHODS:
        new=(set(original)|added)-(removed if name=='clear_front' else set())
        dest=out/name/split/key/'reconstruction.zip'
        dest.parent.mkdir(parents=True,exist_ok=True)
        write_pack(dest,{c:original.get(c,'minecraft:smooth_stone').removeprefix('minecraft:') for c in new},JAVA_26_3_FORMAT)
        decoded=decoded_pack(dest)
        assert set(decoded)==new and all(decoded[c]==original[c] for c in new.intersection(original))
        answer=placement_scores(registered(decoded,transform),before,dict.fromkeys(truth),occupied)
        m=answer['relative_geometry']
        record['methods'][name]={k:m[k] for k in FIELDS}
        save(dest.with_name('cells.json'),dict(local=sorted(decoded),world=sorted(registered(decoded,transform))))
        record['methods'][name].update(pack_sha256=digest(dest),translation=answer['translation'],edits=[len(added),len(removed) if name=='clear_front' else 0])
    return record


def totals(rows,method):
    sums=Counter(); changed=0
    for r in rows:
        m=r['methods'].get(method,r['baseline'])
        sums.update({k:m.get(k,0) for k in FIELDS})
        changed+=int(m.get('edits',[0,0])!=[0,0])
    sums['f1']=2*sums['correct']/(2*sums['correct']+sums['missed']+sums['extras'])
    return dict(frames=len(rows),changed=changed,**sums)


def verify_release(rows, out, report):
    control=read(checked(suite.CONTROL/'cases.json',read(suite.CONTROL/'summary.json')['cases_sha256']))
    tested={(r['split'],r['id']):r for r in rows}
    archive=report/'exports.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
        for method in METHODS:
            path=out/method
            manifest=[]
            for original in control:
                split,key=original['split'],original['id']
                row=tested[split,key]
                stats=row['methods'].get(method)
                m=dict(row['baseline'],target=original['relative_geometry']['target'],placed=original['relative_geometry']['placed'])
                result=dict(split=split,id=key,rgb_sha256=row['rgb_sha256'],reference_hashes=original['reference_hashes'],
                            input_hashes=original['input_hashes'],pack_sha256=original['pack_sha256'],
                            control_pack_sha256=original['pack_sha256'],translation=original['translation'],relative_geometry=m)
                if stats:
                    m.update({k:stats[k] for k in FIELDS})
                    m['placed']=stats['correct']+stats['extras']
                    result.update(pack_sha256=stats['pack_sha256'],translation=stats['translation'])
                    for filename in ('reconstruction.zip','cells.json'):
                        z.write(path/split/key/filename,f'{method}/{split}/{key}/{filename}')
                manifest.append(result)
            save(path/'cases.json',manifest)
            save(path/'summary.json',dict(cases_sha256=digest(path/'cases.json')))
            for filename in ('cases.json','summary.json'):
                z.write(path/filename,f'{method}/{filename}')
            for rr in (r for r in rows if r['split']=='shelter' and method in r['methods']):
                for filename in ('reconstruction.zip','cells.json'):
                    z.write(path/'shelter'/rr['id']/filename,f'{method}/shelter/{rr["id"]}/{filename}')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for row in rows:
            for method,stats in row['methods'].items():
                source=checked(out/method/row['split']/row['id']/'reconstruction.zip',stats['pack_sha256'])
                assert source.read_bytes()==z.read(f"{method}/{row['split']}/{row['id']}/reconstruction.zip")
    compare_priority(suite.CONTROL,{m:out/m for m in METHODS},report/'priority.json',
                     HERE/'benchmark/reviews/strip-orientation-01/sliver-counts.json')
    return digest(archive)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('runs/local-placement'))
    ap.add_argument('--report',type=Path,default=Path('benchmark/reviews/facade-local-placement-01'))
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    sha=read(CANDIDATE/'summary.json')['cases_sha256']
    proposals={(r['split'],r['id']):r for r in read(checked(CANDIDATE/'cases.json',sha))}
    jobs=list(suite.jobs([*framed.SIZES,'shelter']))
    assert len(jobs)==len(proposals)==204
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        res=list(pool.map(case,[(*j,proposals[j[0],j[1]['id']],args.out) for j in jobs]))
    args.report.mkdir(parents=True,exist_ok=True)
    save(args.report/'cases.json',res)
    cohorts={s:[r for r in res if r['split']==s] for s in [*framed.SIZES,'shelter']}
    cohorts['fixed']=[r for r in res if r['split'] in framed.EVALUATION]
    answer=dict(surface_cases_sha256=sha,cases_sha256=digest(args.report/'cases.json'),methods={m:{s:totals(rows,m) for s,rows in cohorts.items()} for m in ('baseline',*METHODS)})
    answer['exports_sha256']=verify_release(res,args.out,args.report)
    retired=read(HERE/'benchmark/reviews/visible-error-budget-01/verification.json')['retired_texture_controls']
    answer['historical_texture_controls']=retired
    assert retired['frames']==8 and retired['baseline_placements']==0
    for groups in answer['methods'].values():
        fixed=groups['fixed']
        groups['historical_112_f1']=2*fixed['correct']/(2*fixed['correct']+fixed['extras']+fixed['missed']+retired['visible_targets'])
    save(args.report/'summary.json',answer)
    print({k:{m:answer['methods'][m][k] for m in METHODS} for k in ('train','fixed','extra')},flush=True)


if __name__=='__main__':
    main()

