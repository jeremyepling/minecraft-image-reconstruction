"""Score changed exports on facade-priority targets while retaining every extra placement."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark import framed_depth_eval as framed
from benchmark.facade_visibility import rates
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_worst_stages import read_view
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked,digest,read,save


def records(folder):
    summary=read(folder/'summary.json')
    return {r['id']:r for r in read(checked(folder/'cases.json',summary['cases_sha256']))}


def compare(control,candidates,out,baseline_file):
    original=records(control)
    measured=read(baseline_file)
    audits={r['id']:r for r in measured['frames']}
    sources={name:records(folder) for name,folder in candidates.items()}
    totals={name:Counter() for name in sources}
    frames=[]
    for s,f,inputs in framed.cases(framed.EVALUATION):
        key=f['id']
        before=original[key]
        audited=audits[key]
        m=before['relative_geometry']
        assert before['pack_sha256']==audited['pack_sha256'] and before['rgb_sha256']==audited['rgb_sha256']
        optional_hits,optional_misses,*_=audited['stats']['face_4_horizontal']
        reference=None
        for name,manifest in sources.items():
            after=manifest[key]
            assert after['reference_hashes']==before['reference_hashes'] and after['rgb_sha256']==before['rgb_sha256']
            assert after['control_pack_sha256']==before['pack_sha256']
            tallies=totals[name]
            delta=Counter()
            if after['pack_sha256']!=before['pack_sha256']:
                if reference is None:
                    _,_,_,_,_,labels,ids,truth,_,placed,_=read_view(control,s,f,inputs,before)
                    optional=horizontal_slivers(labels,ids)
                    reference=truth,placed,optional
                truth,placed,optional=reference
                folder=candidates[name]/s/key
                cells=read(folder/'cells.json')
                pack=checked(folder/'reconstruction.zip',after['pack_sha256'])
                assert set(decoded_pack(pack))==set(map(tuple,cells['local']))
                current=framed.spacing.shift(set(map(tuple,cells['world'])),after['translation'])
                assert len(current&truth)==after['relative_geometry']['correct']
                gain,loss=current-placed,placed-current
                delta.update(required_gained=len((gain&truth)-optional),required_lost=len((loss&truth)-optional),
                             optional_gained=len(gain&optional),optional_lost=len(loss&optional),
                             extras_added=len(gain-truth),extras_removed=len(loss-truth))
                frames.append(dict(id=key,method=name,changes=dict(delta),pack_sha256=after['pack_sha256']))
            hits=m['correct']-optional_hits+delta['required_gained']-delta['required_lost']
            misses=m['missed']-optional_misses-delta['required_gained']+delta['required_lost']
            extras=m['extras']+delta['extras_added']-delta['extras_removed']
            assert extras==after['relative_geometry']['extras']
            tallies.update(delta)
            tallies.update(correct=hits,missed=misses,extras=extras,views=1)
    result={}
    for name,tally in totals.items():
        assert tally['views']==104
        result[name]=dict(tally,f1=rates(tally['correct'],tally['missed'],tally['extras'])['f1'],
                          cases_sha256=digest(candidates[name]/'cases.json'))
    save(out,dict(sliver_audit_sha256=digest(baseline_file),baseline_cases_sha256=digest(control/'cases.json'),
                  methods=result,changed=frames))
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,default=Path('../visible-error-budget/runs/primary-production'))
    p.add_argument('--audit',type=Path,default=Path('benchmark/reviews/strip-orientation-01/sliver-counts.json'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('runs',nargs='+',help='Name=folder for each candidate')
    args=p.parse_args()
    compare(args.control,{v.split('=',1)[0]:Path(v.split('=',1)[1]) for v in args.runs},args.out,args.audit)
