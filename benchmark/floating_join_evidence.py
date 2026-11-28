"""Evaluate source-only perpendicular wall-join evidence on frozen floating wall packets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark import expanded_scores, relative_face_assembly as solver
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import cases
from benchmark.floating_cloud_replay import check_handoff
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_scoring import evaluate_relative_voxels, yaw_from_anchor_normal
from engine.export import JAVA_26_3_FORMAT, write_pack
from engine.surface_perimeter import _native_alpha


FRESH = WORKTREES / 'floating-fresh-validation/runs/floating-fresh'
PRIOR = WORKTREES / 'expanded-wall-rulers/runs/wall-rulers/source_material'
CROP = WORKTREES / 'crop-ceiling-expanded/runs/native-crop-ceiling'
IDEAL = WORKTREES / 'expanded-wall-grids/runs/wall-grids/source-normal'
OUTPUT = ROOT / 'runs/floating-join-evidence'
ARMS = ('existing', 'two_distinct', 'separated', 'supported', 'bypass')


def evidence(join, charts, masks, depth):
    """Look for two spatially separate agreements or exclusive support across one junction."""
    segments = np.asarray(join['image_edges'])
    centers = segments.mean(axis=1)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    separated = any(np.linalg.norm(centers[a] - centers[b]) >= 1.5 * max(lengths[a], lengths[b])
                    for a, b in combinations(range(len(segments)), 2))
    planes = [masks[charts[join[k]]['id']] for k in ('i', 'j')]
    records = []
    for (start, end), length in zip(segments, lengths):
        normal = (end - start)[[1, 0]] * [-1, 1] / length
        points = start + np.linspace(.15, .85, 9)[:, None] * (end - start)
        # Same native pixel band for both masks. Larger cells allow a wider band;
        # cap the distance to avoid reading a nearby edge on a narrow wall.
        near = min(4., max(1.5, length * .12))
        coords = np.floor(points[None, :, :] + np.array([-2, -1, 1, 2])[:, None, None]
                          * near * normal).astype(int)
        x, y = coords[..., 0], coords[..., 1]
        inside = ((x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])).all(axis=0)
        x = x.clip(0, depth.shape[1] - 1)
        y = y.clip(0, depth.shape[0] - 1)
        sides = []
        for alpha, (x0, y0, x1, y1) in planes:
            valid = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            sides.append(valid & alpha[(y-y0).clip(0,y1-y0-1), (x-x0).clip(0,x1-x0-1)])
        a, b = sides
        owner = max((inside & a[i] & ~b[i] & b[j] & ~a[j] for i,j in ((1,2),(2,1))),
                    key=np.count_nonzero)
        left2, left, right, right2 = depth[y,x]
        noise = np.maximum(.045 * (np.abs(left) + np.abs(right)) / 2,
                           2 * (np.abs(left-left2) + np.abs(right-right2)))
        continuous = np.abs((2*left-left2) - (2*right-right2)) <= noise
        records.append(dict(length=float(length), opposite=int(owner.sum()),
                            continuous_opposite=int((owner & continuous).sum())))
    # One tight, physically supported pixel edge can connect small components.
    # For a longer repeated junction require one segment with genuine side ownership.
    supported = any(r['continuous_opposite'] >= 7 and
                    (separated or join['error'] <= min(1., .07 * r['length'])) for r in records)
    return dict(chart_ids=[charts[join[k]]['id'] for k in ('i','j')], shift=join['shift'],
                edges=join['edges'], error=join['error'], segments=join['image_edges'],
                separated=separated, supported=supported, observations=records)


def inputs(case, group):
    name = case['id']
    if group == 'fresh':
        tree, handoffs = FRESH/'source_material', FRESH/'ordinary'
    elif group == 'ideal':
        tree, handoffs = IDEAL, CROP/'native'
    else:
        tree, handoffs = PRIOR, CROP/'source'
    depth_root = WORKTREES/'full-expanded-controls/runs/expanded-baseline-01'/case['dataset']/name
    if (depth_root/'handoffs/depth.npz').is_file():
        assert read(depth_root/'receipt.json')['source_sha256'] == case['rgb_sha256']
        depth_file = depth_root/'handoffs/depth.npz'
    else:
        depth_file = (FRESH if group == 'fresh' else CROP)/'fields'/case['dataset']/name/'depth.npz'
    return tree, handoffs/'sky'/name/'source-handoff', depth_file


def reference(case):
    if case['dataset'] != 'visible-builds-v2':
        return expanded_scores.load_reference(case)
    proof = next(p for p in read(ROOT/'training/framing-audit-v2.json')['frames'] if p['id'] == case['id'])
    assert proof['accepted'] and proof['rgb_sha256'] == case['rgb_sha256']
    meta = Path(case['frame'])/'frame.json'
    def framed(path):
        data = read(path)
        return data | dict(framing=proof) if path == meta else data
    with patch.object(expanded_scores, 'read', framed):
        return expanded_scores.load_reference(case)


def options(joins, measurements, exhaustive):
    all_indices = tuple(range(len(joins)))
    named = dict(existing=all_indices, two_distinct=tuple(i for i in all_indices if joins[i]['edges'] >= 2),
                 separated=tuple(i for i in all_indices if measurements[i]['separated']),
                 supported=tuple(i for i in all_indices if measurements[i]['supported']), bypass=())
    if exhaustive:
        # Avoid exponential expansion on dense ideal charts or future packet sets.
        subsets = (s for n in range(len(joins)+1) for s in combinations(all_indices,n)) if len(joins)<=8 else (
            tuple(j for j in all_indices if j!=i) for i in all_indices)
        for subset in subsets:
            if subset not in named.values():
                named['keep_' + '_'.join(map(str,subset))] = subset
    return named


def run(job):
    case, group = job
    name = case['id']
    frame = Path(case['frame'])
    tree, handoff, depth_file = inputs(case, group)
    packet_file = tree/'charts/sky'/name/'charts.json'
    packet = read(packet_file)
    image = frame/'rgb.png'
    assert digest(image) == packet['source_sha256'] == case['rgb_sha256']
    provenance = check_handoff(packet, handoff, case['rgb_sha256'])
    with Image.open(image) as src:
        size = src.size
    with np.load(depth_file, allow_pickle=False) as data:
        depth = np.asarray(Image.fromarray(data['value'].astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    raw = packet['charts']
    oriented, hint = solver.oriented(raw)
    joins, _ = solver.propose_joins(oriented, min_edges=1, edge_pixels=2.5)
    coplanar = solver.coplanar_joins(oriented, .5)
    wanted = {oriented[j[k]]['id'] for j in joins for k in ('i','j')}
    masks = {}
    for part in read(handoff/'surfaces.json')['planes']:
        if part['id'] in wanted:
            alpha, box = _native_alpha(part['support'], size)
            masks[part['id']] = np.asarray(alpha)>0, box
    measured = [evidence(join, oriented, masks, depth) for join in joins]
    candidates = options(joins, measured, group != 'ideal')
    folder = OUTPUT/group/name
    # All source choices are frozen before decoding and scoring. Reuse a decoded
    # export only when every local placement, ID and assignment is identical.
    truth = reference(case)
    yaw = yaw_from_anchor_normal(hint['normal_camera'], solver.axis_vector(hint['local_normal_axis']), truth.pose) if hint else None
    scores, aligned, selected, unique, cached = {}, {}, {}, {}, {}
    for arm, keep in candidates.items():
        if keep not in cached:
            accepted = [dict(joins[i]) for i in keep]
            with patch.object(solver,'propose_joins',return_value=(accepted,0)), \
                 patch.object(solver,'coplanar_joins',return_value=[dict(c) for c in coplanar]):
                assembled = solver.assemble(raw, min_edges=1, edge_pixels=2.5, coplanar_pixels=.5)
            main = assembled['groups'][0] if assembled['groups'] else dict(blocks={}, assigned=set(),origins={})
            key = (frozenset(main['blocks'].items()), frozenset(main['assigned']))
            if key not in unique:
                export = folder/(arm+'.zip')
                folder.mkdir(parents=True, exist_ok=True)
                write_pack(export,{p:m.removeprefix('minecraft:') for p,m in main['blocks'].items()},JAVA_26_3_FORMAT)
                local = decoded_pack(export)
                assert local == {p:'minecraft:'+m.removeprefix('minecraft:') for p,m in main['blocks'].items()}
                quarters = 0 if not local else yaw['yaw_quarters'] if yaw else None
                score, mapping = evaluate_relative_voxels(local,truth,yaw_quarters=quarters,relaxed_yaw=True,assigned=main['assigned'])
                prior = tree/'assembly/connected-atlas/sky'/name
                if arm == 'existing' and (prior/'score.json').is_file():
                    previous = read(prior/'score.json')
                    assert previous['chart_sha256'] == digest(packet_file)
                    assert previous['pack_sha256'] == digest(prior/'reconstruction.zip')
                    assert local == decoded_pack(prior/'reconstruction.zip') and score == previous['score']
                unique[key] = dict(score=score['fixed'] or score['relaxed_yaw'], pack=export.name,
                                   pack_sha256=digest(export)), mapping
            cached[keep] = (*unique[key], dict(origins=main['origins'], components=len(assembled['groups']),
                                               joins=len(assembled['joins']), conflicts=len(assembled['conflicts'])))
        scores[arm], aligned[arm], selected[arm] = cached[keep]
    original = aligned['existing'].keys()
    for arm, mapping in aligned.items():
        added, removed = mapping.keys()-original, original-mapping.keys()
        gain, loss = added & truth.truth.keys(), removed & truth.truth.keys()
        selected[arm] = selected[arm] | dict(gained=len(gain), lost=len(loss),
            extras_added=len(added-gain), extras_removed=len(removed-loss))
    winners = {metric:max(scores,key=lambda a:scores[a]['score'][metric]['strict']['f1']) for metric in ('geometry','scores')}
    union = set().union(*(coords.keys() for coords in aligned.values()))
    result = dict(id=name, dataset=case['dataset'], layout=case['layout'], split=case['split'], source_sha256=digest(image),
                  chart_sha256=digest(packet_file), depth_sha256=digest(depth_file), reference_sha256=truth.hashes, **provenance,
                  proposals=measured, coplanar=len(coplanar), options=candidates, selected=selected,
                  scores=scores, oracle=winners, missing_from_every_option=len(truth.truth.keys()-union))
    save(folder/'result.json.gz',result)
    print(group,name,len(joins),'proposals,',len(unique),'different exports',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('prior','fresh','ideal'))
    parser.add_argument('--id', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rows = read(FRESH/'manifest.json')['cases'] if args.group=='fresh' else list(cases())
    jobs = [(case,args.group) for case in rows if not args.id or case['id'] in args.id]
    if args.id and len(jobs)!=len(set(args.id)):
        raise ValueError('Requested frame missing from manifest')
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))


if __name__ == '__main__':
    main()
