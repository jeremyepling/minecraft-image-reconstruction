"""Score one material choice per group against the eligible reference blocks in each picture."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path

from benchmark.group_metrics import read


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(samples, methods, minimum):
    count = Counter()
    material = {name: Counter() for name in methods}
    for frame, ref_frame in samples:
        observations = {r["id"]: r for r in frame["faces"]}
        refs = {r["id"]: r for r in ref_frame["faces"]}
        assert set(refs).issubset(observations)
        blocks = defaultdict(list)
        for idx, ref in refs.items():
            blocks[tuple(ref["xyz"])].append(idx)
        groups = {g["id"]:g for g in frame["groups"]}
        partition = [idx for g in groups.values() for idx in g["face_ids"]]
        assert len(partition)==len(observations) and set(partition)==set(observations)
        predicted_pairs = {tuple(sorted(ids)) for g in groups.values() for ids in combinations(g["face_ids"],2)}
        true_pairs = {tuple(sorted(ids)) for items in blocks.values() for ids in combinations(items,2)}
        scored_pairs = {pair for pair in predicted_pairs if all(i in refs for i in pair)}
        calls = 0
        for group in groups.values():
            selected = group["selected_id"]
            for idx in group["face_ids"]:
                assert observations[idx]["selected_id"]==selected and observations[idx]["group_id"]==group["id"]
            supported = [idx for idx in group["face_ids"] if observations[idx]["effective_width_px"] >= minimum]
            assert (selected in supported if supported else selected is None)
            assert all((pick is not None)==bool(supported) for pick in group["material"].values())
            assert set(methods)==set(group["material"])
            calls += selected is not None
        assert calls==frame["matcher_calls_per_method"]
        count.update(frames=1, populated_frames=bool(observations), faces=len(observations), matched_faces=len(refs),
                     blocks=len(blocks), predicted_groups=len(groups), ranked_groups=calls,
                     predicted_pairs=len(predicted_pairs), true_pairs=len(true_pairs), true_positive=len(scored_pairs&true_pairs),
                     false_positive=len(scored_pairs-true_pairs), false_negative=len(true_pairs-scored_pairs),
                     unmatched_pairs=len(predicted_pairs-scored_pairs))
        for position, ids in blocks.items():
            expected = refs[ids[0]]["block_id"]
            assert all(refs[i]["block_id"]==expected for i in ids)
            eligible = [i for i in ids if observations[i]["effective_width_px"] >= minimum]
            count.update(eligible_blocks=bool(eligible),under_five_only_blocks=not eligible,
                         eligible_faces=len(eligible),under_five_faces=len(ids)-len(eligible))
            if not eligible:
                continue
            for method in methods:
                def correct(idx):
                    return groups[observations[idx]["group_id"]]["material"][method]==expected
                material[method].update(correct_blocks=all(correct(i) for i in eligible),
                    correct_eligible_faces=sum(correct(i) for i in eligible),
                    correct_inherited_thin_faces=sum(correct(i) for i in ids if i not in eligible))
    return dict(coverage=dict(count),material=material)


def main(args):
    predictions=read(args.predictions)
    digests=read(args.inputs/'input-digests.json')
    by_group=defaultdict(list)
    truths={}; fixture={}
    for name in ('faces','diagnostic-faces','truth','diagnostic-truth'):
        path=args.inputs/(name+'.json')
        assert sha(path)==digests[path.name]
        if name.endswith('faces'):
            assert digests[path.name]==predictions['sha256'][path.name]
        records=fixture if name.endswith('faces') else truths
        records.update({(f['kind'],f['key']):f for f in read(path)['frames']})
    seen=set()
    for frame in predictions['frames']:
        pair=frame['kind'],frame['key']
        assert pair not in seen
        seen.add(pair)
        frozen=fixture[pair]
        assert frozen['mask_sha256']==frame['mask_sha256'] and frozen['sha256']==frame['sha256']
        assert frozen['mask_variant']==frame['mask_variant']=='sky'
        assert [(r['id'],r['quad']) for r in frozen['faces']]==[(r['id'],r['quad']) for r in frame['faces']]
        for k in (pair[0],pair[0]+(' development' if '/dev-' in pair[1] else ' regression')):
            by_group[k].append((frame,truths[pair]))
    assert seen==set(fixture)==set(truths)
    save = dict(source_sha256=sha(args.predictions), minimum_width=predictions['minimum_width'],
                grouping_unit='one physical block per picture', under_five_only_blocks='excluded',
                fixtures=digests, scores={k:summarize(v,predictions['methods'],predictions['minimum_width']) for k,v in by_group.items()})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(save,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','predictions','out'):parser.add_argument(name,type=Path)
    main(parser.parse_args())
