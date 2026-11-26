"""Compare exported proposals to the cebbffa control using visible geometry in each input view."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import Path as ZipPath, ZipFile

from benchmark import canopy_fixture_metrics as openings
from benchmark import relative_canopy_spacing as spacing
from benchmark import visible_depth_native as native
from benchmark import visible_depth_transfer as transfer
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.relative_geometry_replay import FIELDS, references
from benchmark.texture_camera_eval import DATA, ROOT, STABLE, digest, frames, read, save
from benchmark.visible_recovery_report import SIZES, summarize


REVIEWS = ROOT / 'benchmark/reviews'
MAIN_ARCHIVE = REVIEWS / 'visible-recovery-01/scores.zip'
EXTRA_REVIEW = REVIEWS / 'visible-recovery-extra-views-01'
EXTRA_INPUTS = STABLE.parent / 'minecraft-image-reconstruction-worktrees/visible-recovery-extra-views/runs/extra-views'
EVALUATION = ('validation', 'test', 'regression', 'native', 'old', 'fresh')


def records(folder, ids):
    cases = (folder/'cases.json').read_bytes()
    manifest = json.loads((folder/'manifest.json').read_bytes())
    assert sha256(cases).hexdigest() == manifest['cases_sha256'], folder
    rows = json.loads(cases)
    index = {r['id']:r for r in rows}
    assert index.keys() == ids and len(rows) == len(ids), folder
    return index, manifest


def decode(folder, row, hashes, transform):
    assert all(row[k] == v for k,v in hashes.items()), row['id']
    base = folder/row['id']
    pack = (base/'reconstruction.zip').read_bytes()
    assert sha256(pack).hexdigest() == row['pack_sha256'], row['id']
    local = set(decoded_pack(BytesIO(pack)))
    coords = json.loads((base/'cells.json').read_bytes())
    assert local == set(map(tuple,coords['local'])) and (transform is not None or not local), row['id']
    world = registered(local, transform)
    assert world == set(map(tuple,coords['world'])), row['id']
    return world


def canopy_report(rows):
    return dict(frames=len(rows), visible_roof=sum(r['spacing']['visible_roof_cells'] for r in rows),
                anchored_roof=sum(r['spacing']['anchored_roof_cells'] for r in rows),
                target_triples=sum(r['spacing']['target_triples'] for r in rows),
                matching_triples=sum(r['spacing']['matching_triples'] for r in rows),
                exact_widths=sum(r['spacing']['observed_gap_width']==r['spacing']['target_gap_width'] for r in rows),
                unresolved_widths=sum(r['spacing']['observed_gap_width'] is None for r in rows),
                clear_rays=sum(r['clear_rays'] for r in rows), filled_clear_rays=sum(r['filled_clear_rays'] for r in rows))


def compare(args):
    fixed_roofs = {r['id']: (r['id'],r['current']) for r in read(REVIEWS/'visible-recovery-01/spacing/cases.json')['cases']}
    extra_roofs = {r['id']: (r['previous_probe'],r['candidate']) for r in read(EXTRA_REVIEW/'roofs.json')}
    roof_specs = read(openings.PROBES)
    results, manifests = [], {}
    with ZipFile(MAIN_ARCHIVE) as main, ZipFile(EXTRA_REVIEW/'scores.zip') as extra:
        for split in args.splits:
            alternate = split=='extra'
            source = 'old' if alternate else split
            depth = args.runs/'visible-depth-rounds'
            inputs = {'extra':args.extra_inputs,'old':depth/'transfer','fresh':args.runs/'canopy-placement/confirmation'}.get(split)
            if inputs is not None:
                sample = read(inputs/'screen.json')['selected']
            elif split=='native':
                sample = native.rows()
            else:
                sample = frames(args.data,split)
            ids = {f['id'] for f in sample}
            assert len(ids)==len(sample)==(20 if alternate else SIZES[split]), split
            old_folder = ZipPath(extra,'candidate/') if alternate else ZipPath(main,split+'/')
            new_folder = args.extra_predictions if alternate else (args.predictions/split if args.predictions else None)
            if (args.predictions or args.extra_predictions) and new_folder is None:
                raise ValueError(f'Provide candidate predictions for the requested {split} cohort')
            old_rows, frozen_manifest = records(old_folder,ids)
            new_rows, trial_manifest = records(new_folder,ids) if new_folder else (old_rows,frozen_manifest)
            manifests[split] = {k:{f:v for f,v in m.items() if f in ('cases_sha256','revision','source_dirty',
                'selection_sha256','candidate_sha256','adapter_sha256')} for k,m in [('control',frozen_manifest),('candidate',trial_manifest)]}
            if inputs:
                selection = digest(inputs/'screen.json')
                assert all(m['selection_sha256']==selection for m in (frozen_manifest,trial_manifest)),split
            for f in sample:
                key = f['id']
                truth, occupied, transform, hashes = references(source,f,inputs,args.data,depth,
                    dict(native=args.native_fixtures,old=transfer.BUILDS,fresh=transfer.BUILDS))
                hashes['rgb_sha256']=f['files']['rgb.png']
                before = decode(old_folder,old_rows[key],hashes,transform)
                after = decode(new_folder,new_rows[key],hashes,transform) if new_folder else before
                scored = placement_scores(after,before,dict.fromkeys(truth),occupied)
                for field in ('metrics','relative_geometry'):
                    counts = scored[field]
                    del counts['joint_correct']  # This audit is geometry-only.
                    assert all(counts[k]==new_rows[key][field][k] for k in FIELDS), (split,key)
                    assert counts['correct']-counts['gained']+counts['lost']==old_rows[key][field]['correct']
                    assert counts['extras']-counts['extra_added']+counts['extra_removed']==old_rows[key][field]['extras']
                    assert counts['hidden_extras']-counts['hidden_added']+counts['hidden_removed']==old_rows[key][field]['hidden_extras']
                assert list(scored['translation'])==new_rows[key]['translation'] and list(scored['baseline_translation'])==old_rows[key]['translation']
                row = dict(split=split,id=key,files=f['files'],inputs=hashes,control_pack_sha256=old_rows[key]['pack_sha256'],
                           candidate_pack_sha256=new_rows[key]['pack_sha256'],**scored)
                probe = (extra_roofs if alternate else fixed_roofs).get(key)
                if probe:
                    spec_key, expected = probe
                    spec = roof_specs[spec_key]
                    baseline = spacing.score(before,truth,spec)
                    assert json.loads(json.dumps(baseline))=={k:v for k,v in expected.items() if k!='pack_sha256'}
                    rights = openings.score(f,after,spec)
                    row['canopy'] = dict(spacing=spacing.score(after,truth,spec),clear_rays=len(rights['rays']),
                                         filled_clear_rays=sum(r['foreground_filled'] for r in rights['rays']))
                results.append(row)
    groups = {s:[r for r in results if r['split']==s] for s in args.splits}
    if set(EVALUATION)<=set(args.splits):
        groups['evaluation'] = [r for r in results if r['split'] in EVALUATION]
    if 'evaluation' in groups and 'extra' in groups:
        assert not {r['inputs']['rgb_sha256'] for r in groups['evaluation']} & {r['inputs']['rgb_sha256'] for r in groups['extra']}
    report = dict(control='cebbffa',primary='pooled visible voxel F1 after one whole-build translation',
                  archives_sha256={p.parent.name:digest(p) for p in (MAIN_ARCHIVE, EXTRA_REVIEW/'scores.zip')},
                  manifests=manifests,groups={s:summarize(rows) for s,rows in groups.items()},
                  canopies={s:canopy_report([r['canopy'] for r in rows if 'canopy' in r])
                            for s,rows in groups.items() if any('canopy' in r for r in rows)})
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'cases.json',results)
    report['cases_sha256']=digest(args.out/'cases.json')
    save(args.out/'summary.json',report)
    for group in ('train','evaluation','extra'):
        if group in groups:
            m=report['groups'][group]
            print(group, len(groups[group]), m['correct'], m['extras'], f"{m['f1']:.4%}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', nargs='+', choices=(*SIZES,'extra'), default=(*SIZES,'extra'))
    p.add_argument('--predictions',type=Path,help='Root of candidate placement outputs with cohort folders')
    p.add_argument('--extra-predictions',type=Path,help='Additional-view candidate folder containing cases.json')
    p.add_argument('--extra-inputs',type=Path,default=EXTRA_INPUTS)
    p.add_argument('--runs',type=Path,default=STABLE/'runs')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--native-fixtures',type=Path,default=native.BUILDS,
                   help='Archived native-12 sources. Transfer suites use MCIR_NATIVE_TRANSFER_FIXTURES.')
    p.add_argument('--out',type=Path,required=True)
    compare(p.parse_args())
