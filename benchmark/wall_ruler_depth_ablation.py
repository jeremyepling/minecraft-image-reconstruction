"""Separate ranked-depth confirmations from new-depth choices on frozen native rulers."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import argparse

from benchmark import local_wall_ruler_exports as exports
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.expanded_baseline import save
from benchmark.local_wall_rulers import CONTROLS, ROOT, WORK, read
from benchmark.local_wall_ruler_summary import inventory
from benchmark.typed_structures import pooled_scores
import numpy as np


ORIGINAL = ROOT / 'benchmark/reviews/local-wall-rulers-01'
PRIOR_RUNS = ROOT.parent / 'local-atlas-wall-rulers/runs'
OUT = ROOT / 'runs/wall-ruler-depth-ablation'
REVIEW = ROOT / 'benchmark/reviews/wall-ruler-depth-ablation-01'
CASTLE = 'test-castle-01-ashlar-high-right'
PRIMARY = ('ordinary', 'ranked', 'new_depth', 'whole')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def signatures(row, whole):
    ranked, changed, union = [], [], []
    for record in row['records']:
        fit = record['arms']['native_single']
        if fit['accepted'] and fit['legal_plane']:
            option = (record['surface'], record['axis'], fit['level'])
            union.append(option)
            (ranked if fit['level'] == record['baseline'] else changed).append(option)
    assert union == [tuple(v) for v in whole['signature']]
    result = dict(ordinary=(), ranked=tuple(ranked), new_depth=tuple(changed), whole=tuple(union))
    if row['id'] == CASTLE:
        old = {e['id']:e['old'] for e in whole['carrier_changes']}
        result['missing_chart'] = tuple(v for v in ranked if old[v[0]] is None)
        result['disagreeing_chart'] = tuple(v for v in ranked if old[v[0]] is not None)
    return result


def build(task):
    exports.OUT = OUT
    exports.build(task)


def receipt(key, sig, source_digest, expected=None):
    folders = ([PRIOR_RUNS/'local-wall-rulers-native-support-01', PRIOR_RUNS/'local-wall-rulers-01']
               if expected or not sig else [OUT])
    for folder in folders:
        for path in sorted((folder/'exports'/key).glob('*/case.json')):
            item = read(path)
            if tuple(tuple(x) for x in item['signature']) != sig:
                continue
            case = item['case']
            raw = (path.parent/'reconstruction.zip').read_bytes()
            assert case['source_sha256'] == source_digest
            assert sha256(raw).hexdigest() == case['pack_sha256']
            if expected and case['pack_sha256'] != expected:
                continue
            return path, case, raw
    raise FileNotFoundError((key, sig, expected))


def run(workers):
    manifest = read(ORIGINAL/'manifest.json')
    assert digest(CONTROLS) == manifest['controls_sha256']
    for name in ['cases.json','exports.json']:
        assert digest(ORIGINAL/name) == manifest['artifacts'][name]
    rows = {r['id']:r for r in read(ORIGINAL/'cases.json')}
    prior = {r['id']:r for r in read(ORIGINAL/'exports.json')['cases']}
    frames = read(CONTROLS)['pilot']['frames']
    entries = exports.entries()
    plans, tasks = {}, []
    for f in frames:
        key = f['id']
        assert digest(WORK/f['rgb']['path']) == rows[key]['source_sha256'] == f['rgb']['sha256']
        plan = signatures(rows[key], prior[key]['arms']['native_single'])
        plans[key] = plan
        for name, sig in plan.items():
            if sig and sig != plan['whole']:
                tasks.append((f,entries[key],sig,[name]))
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(build,tasks))
    cases = []
    for f in frames:
        key = f['id']
        base = f['baseline']
        pack = f['pack']
        path = WORK/pack['path']
        if 'member' in pack:
            with ZipFile(path) as archive:
                raw = archive.read(pack['member'])
        else:
            raw = path.read_bytes()
        base_hash = sha256(raw).hexdigest()
        assert base_hash == pack.get('member_sha256',pack['sha256']) == prior[key]['pack_sha256']
        if 'labels.json' in f['reference']:
            ref = f['reference']['labels.json']
            assert digest(WORK/ref['path']) == ref['sha256']
            blocks = read(WORK/ref['path'])['blocks']
            transform = np.array(base['registration']['transform'])
        else:
            labels, _, transform = fixed.fixed_reference(entries[key])
            blocks = labels['blocks']
        truth = {tuple(b['xyz_world']) for b in blocks}
        before, local = inventory(raw,base,transform)
        noop = None
        if prior[key].get('noop_decoded_equal'):
            noop_path, noop_case, noop_pack = receipt(key, (), f['rgb']['sha256'])
            mapped, noop_local = inventory(noop_pack, noop_case, transform)
            assert mapped == before and noop_local == local
            assert all(noop_case[k] == base[k] for k in ('scores','geometry','translation'))
            noop = dict(receipt=noop_path.relative_to(WORK).as_posix(), receipt_sha256=digest(noop_path),
                        pack_sha256=noop_case['pack_sha256'])
        result = {}
        decoded = {}
        for name, sig in plans[key].items():
            new = before
            case = base
            pack_hash = base_hash
            provenance = {'released_pack':pack['path']}
            if sig:
                expected = prior[key]['arms']['native_single']['pack_sha256'] if sig == plans[key]['whole'] else None
                path, case, payload = receipt(key,sig,f['rgb']['sha256'],expected)
                new, _ = inventory(payload,case,transform)
                pack_hash = case['pack_sha256']
                provenance = {'receipt':path.relative_to(WORK).as_posix(),'receipt_sha256':digest(path)}
            decoded[name] = new
            gained, lost = new.keys()-before.keys(), before.keys()-new.keys()
            diff = dict(recovered=len(gained & truth),lost=len(lost & truth),extra_added=len(gained-truth),extra_removed=len(lost-truth))
            result[name] = dict(signature=sig, pack_sha256=pack_hash, provenance=provenance,
                                translation=case['translation'],geometry=case['geometry'],scores=case['scores'],**diff)
            if name == 'whole':
                for k in ['geometry','scores','pack_sha256',*diff]:
                    assert result[name][k] == prior[key]['arms']['native_single'][k]
        same_as_whole = [name for name, geometry in decoded.items() if geometry == decoded['whole']]
        cases.append(dict(same_aligned_blocks_as_whole=same_as_whole,id=key, dataset=f['dataset'],split=f['split'],source_sha256=f['rgb']['sha256'],noop=noop,arms=result))
    summary = {}
    for group in ('all','fixed-104','visible-builds-v3','decorative-builds-v1','development','transfer'):
        members = [r for r in cases if group in ('all',r['dataset'], 'development' if r['split']=='train' else 'transfer')]
        arms = {}
        for name in PRIMARY:
            vals = [r['arms'][name] for r in members]
            arms[name] = {k:pooled_scores([v[k] for v in vals]) for k in ('geometry','scores')}
            arms[name].update({k:sum(v[k] for v in vals) for k in ('recovered','lost','extra_added','extra_removed')})
        summary[group] = dict(tested=len(members),arms=arms)
    REVIEW.mkdir(parents=True,exist_ok=True)
    save(REVIEW/'results.json',dict(controls_sha256=digest(CONTROLS),source_manifest_sha256=digest(ORIGINAL/'manifest.json'),
                                     source_cases_sha256=digest(ORIGINAL/'cases.json'),source_exports_sha256=digest(ORIGINAL/'exports.json'),
                                     cases=cases,summary=summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    run(parser.parse_args().workers)
