"""Summarize shared-edge source-mask and end-to-end placement replays."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.edge_vote_outline_eval import OUT
from benchmark.texture_camera_eval import digest, read, save


FIXED=('validation','test','regression','native','old','fresh')


def summarize(rows,mode):
    output=Counter(frames=len(rows))
    for row in rows:
        output['multi_block_edges']+=row['runs']
        output['voting_blocks']+=row['run_blocks']
        output['source_edge_candidates']+=row['options']
        method=row['methods'].get('protected' if mode=='protected_stable_counts' else mode)
        if method is None:continue
        if mode=='protected_stable_counts' and method['count_changes']:
            output['images_reverted_on_count_change']+=1
            continue
        output['changed_images']+=1
        output['count_changed_images']+=bool(method['count_changes'])
        output['count_changed_surfaces']+=len(method['count_changes'])
        output['matched_gained']+=len(method['gained_surfaces'])
        output['matched_lost']+=len(method['lost_surfaces'])
        output['images_with_harmed_pixel']+=method['harmed']>0
        output.update({k:method[k] for k in ('paired','background','unmatched','improved','harmed','other','higher_iou','lower_iou','pixels')})
        g=method.get('geometry')
        if g is not None:
            output['placement_replays']+=1
            output['control_mismatches']+=not g['control_equal']
            output['placement_changed_images']+=g['added']+g['removed']>0
            output.update({'placement_'+k:v for k,v in g.get('relative',{}).items() if k in
                         ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')})
            output.update({'priority_'+k:v for k,v in g.get('priority_delta',{}).items()})
    return dict(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--ray',type=Path)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    rows=[r for p in args.inputs for r in read(p)]
    assert len(rows)==len({r['id'] for r in rows})==204
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'cases.json',rows)
    cohorts={'development':('train',),'fixed':FIXED,'alternate':('extra',),'new':('new',),'shelter':('shelter',)}
    summary={k:{m:summarize([r for r in rows if r['split'] in splits],m)
                for m in ('two','three','strong','interior','protected','protected_stable_counts')} for k,splits in cohorts.items()}
    report=dict(baseline='abfc0be',candidate='edge_vote_outline.adjustments',cases_sha256=digest(args.out/'cases.json'),cohorts=summary)
    if args.ray:
        from benchmark.edge_vote_outline_corroborated import WITNESSES
        test=read(args.ray)
        assert {(r['id'],r['rgb'],r['pack']) for r in rows}=={(r['id'],r['rgb'],r['pack']) for r in test}
        save(args.out/'ray-cases.json',test)
        for key,splits in cohorts.items():
            pool=[r for r in test if r['split'] in splits]
            active=[r for r in pool if r.get('accepted')]
            def total(name):
                return sum(r['pixel_grade'][name] for r in active)
            summary[key]['corroborated']=dict(frames=len(pool),source_approved_candidates=sum('ops' in r for r in pool),
                changed_images=len(active),corrected_pixels=total('improved'),previously_correct_pixels_lost=total('harmed'),
                third_surface_pixels=total('other'),iou_down=total('lower_iou'),iou_up=total('higher_iou'),
                paired_delta=total('paired'),background_delta=total('background'),
                changed_exports=sum(bool(r['geometry']['added'] or r['geometry']['removed']) for r in active))
            assert all(r['geometry']['control_equal'] for r in active)
        report.update(ray_cases_sha256=digest(args.out/'ray-cases.json'),two_face_witnesses_sha256=digest(WITNESSES))
    save(args.out/'summary.json',report)
    print(summary['fixed']['protected'])
