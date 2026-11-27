"""Frozen full-block lattice comparison. All reference data is opened after source prediction."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import argparse
import json
import os

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MCIR_IMAGE_ONLY='1')
import numpy as np
from PIL import Image, ImageDraw

from benchmark import expanded_fixed_handoffs as fixed, expanded_scores as scorer, typed_structures as typed
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, MEASURED_REVISION, digest, frames, read, save
from benchmark.full_block_lattice import infer, atlas, FALLBACK
from benchmark.mask_blocks import decoded_pack


BASE = ROOT.parent / 'full-expanded-controls/runs/expanded-baseline-01'
RUN = ROOT / 'runs/full-block-lattice-01'
REVIEW = ROOT / 'benchmark/reviews/full-block-lattice-01'
FIXED = scorer.FIXED_DATASET


@lru_cache(maxsize=1)
def sites():
    return read(typed.REVIEW/'sites.json.gz')


def compact_row(row):
    return {k:row[k] for k in ('geometry','scores','targets','translation','pack_sha256','visible_truth','required_truth','exported','camera_usable')}


def compare(a,b,truth,partial):
    return dict(recovered=len((b.keys()-a.keys()) & truth.keys()),lost=len((a.keys()-b.keys()) & truth.keys()),
                extra_added=len((b.keys()-a.keys())-truth.keys()),extra_removed=len((a.keys()-b.keys())-truth.keys()),
                partial_recovered=len((b.keys()-a.keys()) & partial),partial_lost=len((a.keys()-b.keys()) & partial),
                exact_id_gained=sum(b.get(p)==v and a.get(p)!=v for p,v in truth.items()),
                exact_id_lost=sum(a.get(p)==v and b.get(p)!=v for p,v in truth.items()))


def overlay(image, evidence, path):
    pic = image.copy()
    pen = ImageDraw.Draw(pic)
    for a in evidence.get('anchors',[]):
        pts = [tuple(x) for x in a['quad']]
        pen.line(pts+pts[:1], fill=(45,255,130), width=2)
    pic.thumbnail((900,600))
    pic.save(path,quality=89)


def frame(entry):
    key, dataset = entry['id'], entry['dataset']
    out = RUN/dataset/key
    out.mkdir(exist_ok=True,parents=True)
    if dataset==FIXED:
        image, parts, _, source_camera = fixed.source_inputs(entry)
        job = fixed._jobs()[key]
        status = job[3]['status']
        lines = []
        with ZipFile(typed.REVIEW/'typed-exports.zip') as z:
            pack = z.read(key+'/reconstruction.zip')
        assignment = {tuple(r[:3]):r[4] for r in sites()[key]['rows']}
        assert sha256(pack).hexdigest()==sites()[key]['pack_sha256'] and sites()[key]['source_hash']==entry['rgb_sha256']
    else:
        folder = BASE/dataset/key
        receipt = read(folder/'receipt.json')
        source = Path(entry['frame'])/'rgb.png'
        assert digest(source)==entry['rgb_sha256']==receipt['source_sha256']
        image = Image.open(source).convert('RGB')
        hand = read(folder/'handoffs/placement-input.json.gz')
        parts, source_camera = hand['parts'], hand['camera']
        status = read(folder/'build-manifest.json')['placement']
        source_camera = source_camera or status
        lines = read(folder/'handoffs/mlsd.json.gz')['segments'] if source_camera.get('P') is None else []
        pack = (folder/'reconstruction.zip').read_bytes()
        assert sha256(pack).hexdigest()==receipt['pack_sha256']
        assignment = {tuple(row[x] for x in 'xyz'):row['material_source'] for row in read(folder/'placements.json')['blocks']}
    baseline = {p:b.removeprefix('minecraft:') for p,b in decoded_pack(BytesIO(pack)).items()}
    assert baseline.keys()==assignment.keys()
    partial_ids,P,evidence = infer(image,parts,source_camera or status,status.get('plane_offsets',[]),lines,baseline)
    delta_ids = {p:name for p,name in partial_ids.items() if p not in baseline}
    proposal = baseline | delta_ids
    proposal_status = dict(status, camera_status='supported',P=P,reason=evidence['reason'],
                           camera_source='structure-lattice') if P is not None else status
    evidence.update(new_cells=len(delta_ids), source_sha256=entry['rgb_sha256'])
    save(out/'evidence.json.gz',evidence)
    if evidence.get('anchors'):
        overlay(image,evidence,out/'source.jpg')
    candidate = typed.typed_pack(proposal)[0] if delta_ids else pack
    (out/'proposal.zip').write_bytes(candidate)
    # Inference complete. No native labels or recorded pose used above.
    if dataset==FIXED:
        known = fixed.fixed_reference(entry)
        truth = {tuple(b['xyz_world']):b['block_id'] for b in known[0]['blocks']}
        partial = set()
        def grade(payload, camera, methods):
            return fixed.score_result(entry, fixed.ReplayResult(payload,methods,camera),reference=known)
    else:
        known = scorer.load_reference(entry)
        truth = known.truth
        partial = set().union(*(ps for kind,ps in known.by_kind.items() if kind!='full_cube'))
        def grade(payload,camera,methods):
            return scorer.evaluate_pack(entry,payload,camera,assignment=methods,reference=known)
    base, base_map = grade(pack,status,assignment)
    if delta_ids:
        new_assignment = assignment | {p:'fallback' if b==FALLBACK else 'source-atlas' for p,b in delta_ids.items()}
        alt,alt_map = grade(candidate,proposal_status,new_assignment)
    else:
        alt,alt_map = base,base_map
    choose = bool(delta_ids and evidence.get('accepted'))
    selected,selected_map = (alt,alt_map) if choose else (base,base_map)
    best = alt if alt['geometry']['facade']['f1']>base['geometry']['facade']['f1'] else base
    result = {k:entry[k] for k in ('id','dataset','split','layout','dataset_manifest_sha256','rgb_sha256')}
    diagnostic = {k:v for k,v in evidence.items() if k not in ('anchors','junctions','shared','eye','source_sha256')}
    diagnostic.update(junctions=len(evidence.get('junctions',[])), multi_surface_cells=len(evidence.get('shared',[])),
                      baseline_reason=status.get('atlas_camera_reason') or status.get('reason'))
    result.update(baseline=compact_row(base), source=compact_row(selected), oracle=compact_row(best),
                  proposal=compact_row(alt), source_delta=compare(base_map,selected_map,truth,partial),
                  proposal_delta=compare(base_map,alt_map,truth,partial), evidence=diagnostic,
                  changed=choose, rescued=choose and not base['camera_usable'],grader_sha256=base['reference_sha256'])
    save(out/'result.json',result)
    return key,choose,len(delta_ids)


def aggregate(entries):
    rows = [read(RUN/e['dataset']/e['id']/'result.json') for e in entries if (RUN/e['dataset']/e['id']/'result.json').exists()]
    for row in rows:
        base, proposal = row['baseline'],row['proposal']
        for name,metric in [('oracle','facade'),('strict_oracle','strict')]:
            row[name] = proposal if proposal['geometry'][metric]['f1'] > base['geometry'][metric]['f1'] else base
        row['rescue_only'] = row['source'] if not base['camera_usable'] else base
    summary = {}
    for dataset, n in {FIXED:104,**COHORTS}.items():
        group=[r for r in rows if r['dataset']==dataset]
        if not group:
            continue
        def pool(group):
            item = {}
            for arm in ('baseline','source','proposal','oracle','strict_oracle','rescue_only'):
                item[arm] = {k:typed.pooled_scores([r[arm][k] for r in group]) for k in ('geometry','scores')}
                kinds = defaultdict(Counter)
                for r in group:
                    for name,vals in r[arm]['targets'].items():
                        kinds[name].update(vals)
                item[arm]['targets'] = dict(kinds)
            return item
        item = pool(group)
        item.update(tested=len(group),available=n,splits={split:dict(tested=len(rs),**pool(rs))
                        for split in sorted({r['split'] for r in group}) if (rs:=[r for r in group if r['split']==split])},
                    delta={k:sum(r['source_delta'][k] for r in group) for k in group[0]['source_delta']},
                    rescue_delta={k:sum(r['source_delta'][k] for r in group if r['rescued']) for k in group[0]['source_delta']},
                    heldout={split:dict(tested=len(rs),rescued=sum(r['rescued'] for r in rs),
                        delta={k:sum(r['source_delta'][k] for r in rs if r['rescued']) for k in group[0]['source_delta']})
                        for split in sorted({r['split'] for r in group}) if (rs:=[r for r in group if r['split']==split])},
                    reasons=dict(Counter(r['evidence']['reason'] for r in group)),rescued=sum(r['rescued'] for r in group),
                    source_gate=sum(r['changed'] for r in group),proposed=sum(r['evidence']['new_cells']>0 for r in group),
                    camera_before=sum(r['baseline']['camera_usable'] for r in group),
                    camera_after=sum(r['source']['camera_usable'] for r in group),
                    merged_observations=sum(r['evidence']['multi_surface_cells'] for r in group))
        summary[dataset] = item
    official_file = BASE/'scores/cases.json'
    controls = {(r['dataset'],r['id']):r for r in read(official_file)}
    for r in rows:
        control = controls[r['dataset'],r['id']]
        assert control['source_sha256']==r['rgb_sha256']
        for k in ('pack_sha256','geometry','scores','translation'):
            assert r['baseline'][k]==control[k], (r['id'],k)
    save(REVIEW/'cases.json.gz',rows)
    save(REVIEW/'summary.json',summary)
    save(REVIEW/'manifest.json',dict(base_revision=MEASURED_REVISION,atlas=atlas().provenance,
        implementation={path.name:digest(path) for path in sorted((ROOT/'benchmark').glob('full_block_lattice*.py'))},
        ordinary_control_cases_sha256=digest(official_file),matched_ordinary_controls=len(rows),
        cases_sha256=digest(REVIEW/'cases.json.gz'),fixed_inputs='checked historical pre-inset and released typed pack',
        oracle='per-image facade F1 with ordinary retained on ties; strict_oracle uses strict F1 and keeps ordinary on ties',
        captures=[{k:e[k] for k in ('dataset','id','rgb_sha256','split','dataset_manifest_sha256')} for e in entries]))
    print({name:(r['tested'],r['rescued'],r['baseline']['geometry']['strict']['f1'],r['source']['geometry']['strict']['f1']) for name,r in summary.items()},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cohorts',nargs='+',choices=[FIXED,*COHORTS],default=[FIXED,*COHORTS])
    p.add_argument('--ids',nargs='*')
    p.add_argument('--summary',action='store_true')
    a=p.parse_args()
    entries=(list(fixed.fixed_frames()) if FIXED in a.cohorts else [])+list(frames(SOURCE,[k for k in a.cohorts if k!=FIXED]))
    if a.summary:
        return aggregate(entries)
    todo=[e for e in entries if (not a.ids or e['id'] in a.ids) and not (RUN/e['dataset']/e['id']/'result.json').exists()
          and (e['dataset']==FIXED or (BASE/e['dataset']/e['id']/'receipt.json').exists())]
    if a.workers==1:
        for e in todo:
            print(frame(e),flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            for record in pool.map(frame,todo):
                print(record,flush=True)


if __name__=='__main__':
    main()
