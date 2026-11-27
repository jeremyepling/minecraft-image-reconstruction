"""Check chart choices against decoded coordinates and IDs independent of ZIP metadata."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from benchmark.placement_grid_choice import CONTROL, RUN, fixed, fixed_input, grade, private_charts
from benchmark.expanded_baseline import digest
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import aligned_sites, typed_pack


def semantic(names):
    payload = json.dumps([[*p,names[p]] for p in sorted(names)], separators=(',', ':')).encode()
    return sha256(payload).hexdigest()


def audit(row, entry):
    fixed_set = row['dataset'] == grade.FIXED_DATASET
    base = row['baseline']
    if fixed_set:
        with ZipFile(fixed._release()[2]) as archive:
            local = decoded_pack(BytesIO(archive.read(f"{row['id']}/reconstruction.zip")))
        # The ZIP's inner gzip OS byte differs between CPython 3.12 and 3.13.
        # Re-encode the decoded released inventory in the candidate interpreter.
        canonical, _ = typed_pack({p:n.removeprefix('minecraft:') for p,n in local.items()})
        equivalent = {base['pack_sha256'], sha256(canonical).hexdigest()}
    else:
        local = decoded_pack(CONTROL / row['dataset'] / row['id'] / 'reconstruction.zip')
        equivalent = {base['pack_sha256']}
    base_signature = semantic(local)
    signatures = {base['pack_sha256']: base_signature}
    ref = None
    union = set()
    for v in row['variants']:
        result = v['result']
        key = result['pack_sha256']
        if key in equivalent:
            assert all(result[k] == base[k] for k in ('geometry', 'scores', 'translation'))
            signatures[key] = base_signature
            continue
        selected = v['selections'][0]
        if fixed_set:
            inputs = fixed_input(entry)
            override = {row['options'][i]['id']: row['options'][i]['original'] for i in selected}
            with private_charts(inputs.counts, override):
                replay = fixed.replay_placement(inputs)
            assert sha256(replay.pack).hexdigest() == key
            actual = decoded_pack(BytesIO(replay.pack))
        else:
            folder = RUN / row['dataset'] / row['id'] / ('choice-'+'-'.join(map(str,selected)))
            assert digest(folder/'reconstruction.zip') == key
            actual = decoded_pack(folder/'reconstruction.zip')
        signatures[key] = semantic(actual)
        if actual == local:
            assert all(result[k] == base[k] for k in ('geometry','scores','translation'))
            continue
        if ref is None:
            if fixed_set:
                labels, face_ids, transform = fixed.fixed_reference(entry)
                truth = {tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
                optional = horizontal_slivers(labels, face_ids)
            else:
                import numpy as np
                reference = grade.load_reference(entry)
                baseline_status = grade.read(CONTROL / row['dataset'] / row['id'] / 'build-manifest.json')['placement']
                transform = grade.registration(np.asarray(baseline_status['P']), reference.pose)
                truth, optional = reference.truth, reference.optional
            ref = truth, optional, transform
            reference_points = set(aligned_sites(list(local), transform, base['translation']))
            union.update(reference_points)
        truth, optional, transform = ref
        trial = set(aligned_sites(list(actual), transform, result['translation']))
        union.update(trial)
        gained, lost = trial-reference_points, reference_points-trial
        v['world_delta'] = dict(correct_gained=sorted(gained & truth.keys()), correct_lost=sorted(lost & truth.keys()),
                                extras_added=sorted(gained-truth.keys()), extras_removed=sorted(lost-truth.keys()))
    for item in [v['result'] for v in row['variants']] + [row[n] for n in ('baseline','source','original','oracle')]:
        item['decoded_sha256'] = signatures[item['pack_sha256']]
    if ref:
        truth, optional, _ = ref
        row['required_missing_from_all_choices'] = len(truth.keys()-optional-union)
    else:
        row['required_missing_from_all_choices'] = base['geometry']['facade']['missed']
