"""Draw saved Floating placement changes on their native frontmost source pixels."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageFilter

from benchmark.floating_stage_report import archived_arm, hash_file, read
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_scoring import _yaw
from engine.surface_perimeter import _native_alpha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    return row['dataset'], row.get('panel'), row['id']


def select(frames, candidate):
    """Select actual declines, disjoint swaps and largest paired changes independently."""
    chosen = {}
    def mark(row, reason):
        chosen.setdefault(identity(row), []).append(reason)
    paired = [f for f in frames if f['changes'][candidate]['paired'] is not None]
    dropped=sorted((f for f in paired if f['changes'][candidate]['dropped']),
                   key=lambda f:(not f['changes'][candidate]['disjoint'],-f['changes'][candidate]['paired']['loss'],identity(f)))
    for row in dropped[:2]:
        mark(row,'Disjoint component selection' if row['changes'][candidate]['disjoint'] else 'Dropped parent chart')
    for field, reason in [('loss', 'Most previously correct positions lost'), ('gain', 'Most new correct positions')]:
        if paired:
            winner = min(paired,key=lambda f:(-f['changes'][candidate]['paired'][field],identity(f)))
            if winner['changes'][candidate]['paired'][field]:
                mark(winner,reason)
    return chosen


def aligned_pack(row, frame, arm):
    archived, path = archived_arm(row, arm)
    receipt = read(path)
    pack = path.with_name('reconstruction.zip')
    require(receipt['id']==row['id'] and receipt['source_sha256']==frame['source_sha256'] and
            receipt['reference_sha256']==frame['reference_sha256'],f'Wrong pack/source receipt: {path}')
    require(hash_file(pack)==receipt['pack_sha256']==frame['arms'][arm]['pack_sha256']==archived['pack_sha256'],
            f'Pack differs from stage report: {pack}')
    fixed = receipt['score']['fixed']
    if not fixed:
        raise ValueError(f"No fixed scored registration for {arm} on {row['id']}")
    local = decoded_pack(pack)
    yaw, shift = fixed['yaw_quarters'], fixed['translation']
    aligned = {tuple(v+s for v,s in zip(_yaw(xyz,yaw),shift,strict=True)):block for xyz,block in local.items()}
    saved = read(path.with_name('placements.json'))
    for name, mapping in [('local',local),('aligned',aligned)]:
        require(mapping=={tuple(p):m for p,m in saved[name]},f'{name} positions differ from decoded registration')
    return aligned,receipt,dict(yaw_quarters=yaw,translation=shift,pack_sha256=receipt['pack_sha256'])


def read_frame(row, directory, archives):
    files = ['rgb.png', *row['reference_sha256']]
    if directory:
        frame = directory/row['split']/'frames'/row['id']
        if frame.is_dir():
            return {name:(frame/name).read_bytes() for name in files}
    prefix = 'frames/'+row['id']+'/'
    for path in archives:
        with ZipFile(path) as z:
            if prefix+'rgb.png' in z.namelist():
                return {name:z.read(prefix+name) for name in files}
    raise FileNotFoundError(f"Restore saved frame {row['id']} under --frame-root or supply its --shard.")


def strict_delta(truth,before,after):
    gain=(after.keys()-before.keys())&truth.keys()
    loss=(before.keys()-after.keys())&truth.keys()
    prev={p for p in truth.keys() & before.keys() if truth[p]==before[p]}
    current={p for p in truth.keys() & after.keys() if truth[p]==after[p]}
    return gain,loss,dict(gain=len(gain),loss=len(loss),add=len(after.keys()-before.keys()-truth.keys()),
                          remove=len(before.keys()-after.keys()-truth.keys()),exact_gain=len(current-prev),
                          exact_loss=len(prev-current))


def paint(rgb, mask, color, fraction=.65):
    output=rgb.copy()
    output[mask]=np.rint((1-fraction)*rgb[mask]+fraction*np.array(color)).astype('uint8')
    return output


def render(row, frame, before_arm, after_arm, source_root, directory, archives, out):
    raw = read_frame(row,directory,archives)
    for name,digest in {'rgb.png':frame['source_sha256'],**frame['reference_sha256']}.items():
        require(sha256(raw[name]).hexdigest()==digest,f"Wrong {name} for {row['id']}")
    rgb=np.array(Image.open(BytesIO(raw['rgb.png'])).convert('RGB'))
    labels=json.loads(raw['labels.json'])
    with np.load(BytesIO(raw['labels.npz']),allow_pickle=False) as z:
        front=z['face_id']
    require(front.shape==rgb.shape[:2] and front.min()>=0,'Frontmost owner raster dimensions differ from source')
    faces=labels['faces']
    counts=np.bincount(front.ravel(),minlength=max(f['id'] for f in faces)+1)
    require(all(counts[f['id']]==f['visible_pixels'] for f in faces) and
            counts[1:].sum()==sum(f['visible_pixels'] for f in faces),'Native visible-face ownership mismatch')
    blocks={b['id']:b['block_id'] for b in labels['blocks']}
    truth={tuple(f['xyz_world']):blocks[f['block']] for f in faces}
    a,ar,at=aligned_pack(row,frame,before_arm)
    b,br,bt=aligned_pack(row,frame,after_arm)
    change=frame['changes'][after_arm]
    gain,loss,metrics=strict_delta(truth,a,b)
    require(all(metrics[k]==v for k,v in (change['paired'] or {}).items()),'Replayed site changes differ from stage report')
    for arm,mapping in ((before_arm,a),(after_arm,b)):
        correct=len(mapping.keys()&truth.keys())
        expected=frame['arms'][arm]['scores']['geometry_strict']
        require([correct,len(truth)-correct,len(mapping)-correct]==[expected[k] for k in ('correct','missed','extras')],
                'Registered placement totals differ from report')
    codes=np.zeros(len(counts),'uint8')
    for f in faces:
        xyz=tuple(f['xyz_world'])
        codes[f['id']]=1 if xyz in gain else 2 if xyz in loss else 0
    visible=codes[front]
    name=row['id']
    out.mkdir(parents=True,exist_ok=True)
    image=paint(paint(rgb,visible==1,(0,214,225)),visible==2,(255,66,135))
    Image.fromarray(image).save(out/'placement.png',optimize=True)
    # Only the source mask is highlighted; a fragment absent from the export has no scored world origin.
    component=None
    if change['dropped']:
        surface_path=source_root/'source'/'sky'/name/'source-handoff/surfaces.json'
        require(hash_file(surface_path)==ar['surfaces_sha256']==br['surfaces_sha256'],'Wrong source masks')
        planes={p['id']:p for p in read(surface_path)['planes']}
        selected=np.zeros(front.shape,bool)
        for cid in change['dropped']:
            parent=cid.removeprefix('horizontal-')
            require(parent in planes,f'Dropped chart has no saved source mask: {cid}')
            alpha,(x,y,r,bottom)=_native_alpha(planes[parent]['support'],rgb.shape[1::-1])
            selected[y:bottom,x:r] |= np.asarray(alpha)>0
        picture=paint(rgb,selected,(250,161,49),.32)
        interior=np.asarray(Image.fromarray(selected).convert('L').filter(ImageFilter.MinFilter(5)))>0
        picture[selected & ~interior]=[255,173,46]
        Image.fromarray(picture).save(out/'dropped.png',optimize=True)
        component=dict(surface_sha256=hash_file(surface_path),pixels=int(selected.sum()),dropped=change['dropped'],
                       added=change['added'])
    return dict(id=name,source_sha256=frame['source_sha256'],reference_sha256=frame['reference_sha256'],
                baseline=at,candidate=bt,paired=metrics,component=component,native_pixels=dict(
                    gained=int((visible==1).sum()),lost=int((visible==2).sum())))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('report',type=Path)
    cli.add_argument('conditions',type=Path)
    cli.add_argument('--run-root',type=Path,required=True)
    cli.add_argument('--frame-root',type=Path)
    cli.add_argument('--shard',type=Path,action='append',default=[])
    cli.add_argument('--candidate',default='current')
    cli.add_argument('--case',action='append',default=[],help='Include the indicated case in the visual queue')
    cli.add_argument('--decorative',action='store_true',help='Also generate the separate Decorative queue')
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    report=read(args.report)
    require(hash_file(args.conditions)==report['comparison_sha256'],'Stage report belongs to a different comparison')
    baseline=next(iter(report['names']))
    candidate=args.candidate
    require(candidate in report['names'] and baseline!=candidate,'Unknown candidate')
    rows={identity(r):r for r in read(args.conditions)}
    groups=['Visible Builds']+(['Decorative Builds V1'] if args.decorative else [])
    args.out.mkdir(parents=True,exist_ok=True)
    doc=['# Floating visual review queue','',f"{report['cohort_label'] or 'Saved Floating comparison'}. "
         f"{report['names'][baseline]} compared with {report['names'][candidate]}.",'',
         'Open a PNG at full size to inspect native pixels. Cyan marks gained correct positions and pink marks lost correct positions. '
         'Color follows the recorded frontmost block face. Extra positions have no native owner and appear in the counts. '
         'An amber outline shows the photographed support of a dropped parent chart.','']
    receipts=[]
    for cohort in groups:
        frames=[f for f in report['frames'] if (f['dataset']=='decorative-builds-v1')==(cohort=='Decorative Builds V1')]
        changes=report['cohorts'][cohort]['component_changes'][candidate]
        queue=select(frames,candidate)
        if changes['regressions']:
            worst=changes['regressions'][0]
            match=next(f for f in frames if (f['id'],f['panel'])==(worst['id'],worst['panel']))
            queue.setdefault(identity(match),[]).append('Largest net decline in correct placements')
        by_id={identity(f):f for f in frames}
        for key in by_id:
            if key[2] in args.case:
                queue.setdefault(key,[]).append('Requested visual trace')
        doc+=['## '+cohort,'',f"{len(frames)} scored views. {len(changes['regressions'])} had a net decline in correct placements.",'']
        for key,reasons in queue.items():
            f=by_id[key]
            folder=Path('decorative' if cohort=='Decorative Builds V1' else 'visible')/(f['panel'] or 'original')/f['id']
            label=f['id']
            if any(f['arms'][k]['unresolved'] for k in (baseline,candidate)):
                doc+=['### '+label,'',' / '.join(reasons)+'. Fixed orientation unresolved, so a registered overlay is unavailable.','']
                continue
            entry=render(rows[key],f,baseline,candidate,args.run_root,args.frame_root,args.shard,args.out/folder)
            receipts.append(dict(entry,reasons=reasons,cohort=cohort,panel=f['panel'],folder=folder.as_posix()))
            m=entry['paired']
            doc+=['### '+label,'',' / '.join(reasons)+'.','',
                  f"Correct positions gained {m['gain']}, lost {m['loss']}. Extras added {m['add']}, removed {m['remove']}. "
                  f"Exact material matches gained {m['exact_gain']}, lost {m['exact_loss']}.",'',
                  f"[Open placement overlay]({folder.as_posix()}/placement.png)"]
            doc+=['',f'<a href="{folder.as_posix()}/placement.png"><img src="{folder.as_posix()}/placement.png" '
                  f'width="340" alt="Native placement changes for {label}"></a>']
            if entry['component']:
                doc.append(f"[Dropped chart {', '.join(entry['component']['dropped'])}]({folder.as_posix()}/dropped.png)")
            doc.append('')
    doc+=['## Floating branch scorecard','',
          'Saved scores from the supplied stage report. Queue generation reruns neither inference nor scoring.','',
          '| Cohort and method | Strict / facade placement F1 | Strict / facade position + ID F1 | Strict correct / missed / extra |',
          '|---|---:|---:|---:|']
    for cohort in groups:
        metrics=report['cohorts'][cohort]
        if not metrics['views']:
            continue
        for arm in (baseline,candidate):
            score=metrics['scores'][arm]
            fields=[' / '.join(f"{score[f'{k}_{regime}']['f1_percent']:.2f}%" for regime in ('strict','facade'))
                    for k in ('geometry','scores')]
            counts=' / '.join(f"{score['geometry_strict'][k]:,}" for k in ('correct','missed','extras'))
            doc.append('| '+' | '.join([f"{cohort}, {metrics['views']} views, {report['names'][arm]}",*fields,counts])+' |')
    (args.out/'queue.md').write_text('\n'.join(doc)+'\n',encoding='utf8')
    (args.out/'receipts.json').write_text(json.dumps(dict(stage_report_sha256=hash_file(args.report),
        comparison_sha256=report['comparison_sha256'],cases=receipts),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
