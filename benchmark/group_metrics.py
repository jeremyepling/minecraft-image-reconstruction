"""Score frozen face group predictions against separate, evaluator-only block identities."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path

def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)

def pairs(groups):
    return {tuple(sorted(pair)) for ids in groups for pair in combinations(ids,2)}

def accumulate(result,frame,refs,methods,min_width):
    rows={r['id']:r for r in frame['faces']}
    if len(rows)!=len(frame['faces']): raise ValueError('Duplicate prediction ID')
    blocks=defaultdict(list)
    for idx,ref in refs.items():
        blocks[tuple(ref['xyz'])].append(idx)
    guess=pairs([g['face_ids'] for g in frame['groups']])
    true=pairs(blocks.values())
    comparable={p for p in guess if all(i in refs for i in p)}
    geom=result['geometry']
    geom.update(frames=1,predicted_faces=len(rows),matched_faces=len(refs),blocks=len(blocks),
                predicted_pairs=len(guess),truth_pairs=len(true),true_positive=len(comparable&true),
                false_positive=len(comparable-true),false_negative=len(true-comparable),
                unmatched_pairs=len(guess-comparable),multi_face_blocks=sum(len(ids)>1 for ids in blocks.values()))
    eligible_blocks={pos:[i for i in ids if rows[i]['effective_width_px']>=min_width] for pos,ids in blocks.items()}
    geom.update(under_five_only_blocks=sum(not ids for ids in eligible_blocks.values()),
                eligible_blocks=sum(bool(ids) for ids in eligible_blocks.values()),
                eligible_faces=sum(len(ids) for ids in eligible_blocks.values()),
                under_five_faces=sum(len(blocks[pos])-len(ids) for pos,ids in eligible_blocks.items()))
    for m in methods:
        scores=result['materials'].setdefault(m,{'independent':Counter(),'grouped':Counter(),'change':Counter()})
        for mode in ('independent','grouped'):
            stats=scores[mode]
            for idx,ref in refs.items():
                pred=rows[idx][mode][m]
                stats.update(faces=1,ranked_faces=pred is not None,correct_faces=pred==ref['block_id'])
            for ids in eligible_blocks.values():
                if ids:
                    stats.update(correct_eligible_faces=sum(rows[i][mode][m]==refs[i]['block_id'] for i in ids),
                                 correct_eligible_blocks=all(rows[i][mode][m]==refs[i]['block_id'] for i in ids))
            for ids in blocks.values():
                guesses=[rows[i][mode][m] for i in ids]
                expected=refs[ids[0]]['block_id']
                ok=all(name==expected for name in guesses)
                stats.update(blocks=1,blocks_with_any_correct=expected in guesses,all_faces_correct_blocks=ok,
                             blocks_with_pick=any(v is not None for v in guesses),multi_face_all_correct=ok and len(ids)>1)
        for ids in eligible_blocks.values():
            if ids:
                before,after=(all(rows[i][mode][m]==refs[i]['block_id'] for i in ids) for mode in ('independent','grouped'))
                scores['change'].update(eligible_blocks_gained=after and not before,eligible_blocks_lost=before and not after)
        for idx,ref in refs.items():
            row=rows[idx]
            before=row['independent'][m]==ref['block_id']
            after=row['grouped'][m]==ref['block_id']
            scores['change'].update(faces_gained=after and not before,faces_lost=before and not after,
                                    correct_inherited_under_five=(row['effective_width_px']<min_width and after))
        for ids in blocks.values():
            before,after=(all(rows[i][mode][m]==refs[i]['block_id'] for i in ids) for mode in ('independent','grouped'))
            scores['change'].update(blocks_gained=after and not before,blocks_lost=before and not after)

def main(args):
    data=read(args.predictions); results={}
    fingerprints=read(args.inputs/'input-digests.json')
    def key(f):return f['kind'],f['key']
    truth={}; frozen={}
    for name in ('faces','truth','diagnostic-faces','diagnostic-truth'):
        path=args.inputs/(name+'.json')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==fingerprints[path.name]
        if name.endswith('faces'):assert digest==data['sha256'][path.name]
        group=truth if name.endswith('truth') else frozen
        group.update({key(f):f for f in read(path)['frames']})
    assert set(truth)==set(frozen)=={key(f) for f in data['frames']}
    for frame in data['frames']:
        k=key(frame)
        assert {r['id']:r['quad'] for r in frame['faces']}=={r['id']:r['quad'] for r in frozen[k]['faces']}
        refs={r['id']:r for r in truth[k]['faces']}
        for cohort in (k[0],k[0]+(' development' if '/dev-' in k[1] else ' regression')):
            result=results.setdefault(cohort,dict(geometry=Counter(),materials={}))
            accumulate(result,frame,refs,data['methods'],data['minimum_width'])
    for result in results.values():
        geo=result['geometry']
        geo['pair_precision']=geo['true_positive']/(geo['true_positive']+geo['false_positive']) if geo['true_positive']+geo['false_positive'] else None
        geo['pair_recall']=geo['true_positive']/geo['truth_pairs'] if geo['truth_pairs'] else None
    report=dict(scores=results,source_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                fixtures=fingerprints,block_metric='correct_eligible_blocks requires correct IDs for all independently eligible faces of a block. Blocks with only under-five faces are outside the denominator. all_faces_correct_blocks separately measures full visible coverage, including inherited thin faces.')
    args.out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','predictions','out'):parser.add_argument(name,type=Path)
    main(parser.parse_args())
