"""Verify selected eave repairs on frozen paired replays and fresh public CLI exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import facade_inset_round as suite
from benchmark.eave_patch_audit import REVIEW
from benchmark.edge_vote_outline_eval import baseline_pack, recount
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import digest, read, save
from engine import eave_pinholes, visible_placement
from engine.atlas_count import load_atlas
from engine.surface_perimeter import with_outline


def dimensions(counts):
    return {p['id']:(p['counts'],p['dimensions']['blocks']) for p in counts['surfaces']}


def input_path(job):
    split,frame,_,_=job
    if split=='shelter':
        return suite.SHELTER/frame['directory']/'captures'/(frame['source_id']+'.png')
    return suite.framed.TRAINING_DATA/'data'/split/frame['directory']/'rgb.png'


def replay(job):
    key = job[1]['id']
    image,parts,counts,camera = suite.ordinary_inputs(*job)
    updated, actions = eave_pinholes.refine(image, parts)
    atlas = load_atlas(suite.ATLAS,'26.3') if 'atlas_phase' in counts else None
    before,after = [recount(image,[with_outline(p) for p in source],atlas) for source in (parts,updated)]
    prior_pack, prior = baseline_pack(job)
    original, status = visible_placement.place(image,'26.3',parts,counts=before,camera=camera,atlas_path=suite.ATLAS)
    revised, _ = visible_placement.place(image,'26.3',updated,counts=after,camera=camera,atlas_path=suite.ATLAS)
    assert not any(r['id']==key for r in read(suite.ROOT/'benchmark/reviews/pavement-local-01/summary.json')['changed'])
    assert set(original)==prior, key
    measured=eave_pinholes.measurement_parts(updated)
    assert [p['support'] for p in measured]==[p['support'] for p in parts]
    locked, _ = visible_placement.place(image,'26.3',measured,counts=before,camera=camera,atlas_path=suite.ATLAS)
    assert set(locked)==prior
    return dict(id=key, outline_only_same_export=True, actions=actions, prior_pack_sha256=prior_pack, prior_blocks=len(prior),
                recount_agrees=dimensions(before)==dimensions(after), unrestricted_added=sorted(set(revised)-prior),
                unrestricted_removed=sorted(prior-set(revised)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=('inputs','placements','verify'))
    p.add_argument('--root',type=Path,default=Path('runs/eave-final'))
    args=p.parse_args()
    rows=read(REVIEW/'cases.json')['cases']
    affected={r['id']:r for r in rows if any(e['actions']['fully_surrounded_eave'] for e in r['events'])}
    frozen={r['id']:r for r in read(suite.ROOT/'benchmark/reviews/facade-inset-round-01/combined.json')['rows']}
    jobs=[j for j in suite.jobs([*suite.framed.SIZES,'shelter']) if j[1]['id'] in affected]
    for job in jobs:
        assert not any(frozen[job[1]['id']][k] for k in ('returns','texture','outlines'))
    if args.task=='inputs':
        args.root.mkdir(exist_ok=True,parents=True)
        save(args.root/'tasks.json',[dict(id=j[1]['id'],path=str(input_path(j))) for j in jobs])
    elif args.task=='placements':
        with ProcessPoolExecutor(4) as pool:
            results=list(pool.map(replay,jobs))
        save(REVIEW/'placement.json',results)
        print(results)
    else:
        checks=[]
        for job in jobs:
            key=job[1]['id']
            candidate=args.root/key
            manifest=read(candidate/'build-manifest.json')
            assert manifest['image_sha256']==affected[key]['rgb_sha256'] and manifest['through']=='placement'
            events=manifest['surface_finder']['boundary_settings']['eave_outline']['events']
            targets=[e for e in affected[key]['events'] if e['actions']['fully_surrounded_eave']]
            assert {(e['host'],tuple(e['box']),e['pixels']) for e in events}=={(e['host'],tuple(e['box']),e['pixels']) for e in targets}
            source, parts, counts, _=suite.ordinary_inputs(*job)
            candidate_pack=decoded_pack(candidate/'reconstruction.zip')
            previous_hash,prior=baseline_pack(job)
            assert candidate_pack==dict.fromkeys(prior,'minecraft:smooth_stone'),key
            assert dimensions(counts)==dimensions(read(candidate/'counts.json')),key
            repairs=read(candidate/'surfaces.json')['planes']
            samples={p['id']:p['support'] for p in parts}
            for part in repairs:
                if part.get('measured_eave_support'):
                    assert part['measured_eave_support']==samples[part['id']],key
            checks.append(dict(id=key,pixels=sum(e['pixels'] for e in events),patches=len(events),
                               source_sha256=manifest['image_sha256'],public_counts_unchanged=True,
                               verified_against='frozen export',comparison_pack_sha256=previous_hash,
                               candidate_pack_sha256=digest(candidate/'reconstruction.zip'),exported_blocks=len(candidate_pack)))
        save(REVIEW/'ordinary.json',checks)
        print(checks)


if __name__=='__main__':
    main()
