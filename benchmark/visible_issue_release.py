"""Verify the two promoted stages against decoded exports and two ordinary screenshot builds."""
import argparse
import json
from pathlib import Path

from benchmark import framed_depth_eval as frozen
from benchmark.facade_placement_verify import compare
from benchmark.mask_blocks import decoded_pack
from benchmark.nearby_thin_comparison import TEXTURE_ONLY_RGB
from benchmark.roof_span_results import collect
from benchmark.roof_span_verify import records
from benchmark.texture_camera_eval import checked,digest,read,save


def run(args):
    primary=args.runs/'primary-production'
    reg=records(primary)
    verified={}
    for cohort,n in [('primary',200),('transfer',4),('native',4),('side',12),('elevated',6)]:
        a,b=(args.runs/(cohort+'-'+method) for method in ('combined','production'))
        verified[cohort]=compare(a,b)
        assert verified[cohort]==n
    commands={}
    for tag,key,field in (
        ('canopy','shape-benchmark-26.3--train-canopy-04-left','visible_pillar_blocks'),
        ('relief','signed-relief-26.3--dev--dev-relief-00-oblique-right','panel_blocks_moved')):
        s,= [split for split,frame_id in reg if frame_id==key]
        folder=args.runs/('cli-'+tag)
        meta=read(folder/'build-manifest.json')
        row=reg[s,key]
        assert row['rgb_sha256']==meta['image_sha256']
        actual=decoded_pack(folder/'reconstruction.zip')
        assert actual==decoded_pack(checked(primary/s/key/'reconstruction.zip',row['pack_sha256']))
        stat=meta['placement']
        assert stat[field]>0 and len(actual)==stat['placed_blocks']==stat['strict_blocks']+stat['added_blocks']
        commands[key]=dict(added_or_moved=stat[field],placed=len(actual),pack_sha256=digest(folder/'reconstruction.zip'),
                           screenshot_sha256=meta['image_sha256'])
    paths={'control':args.control,'transfer-control':args.control.parent/'transfer-control'}
    for name in ('combined','production'):
        paths[name]=args.runs/('primary-'+name)
        paths['transfer-'+name]=args.runs/('transfer-'+name)
    for name in ('placed','one_pixel','two_pixels'):
        paths[name]=args.posts/name
        paths['transfer-'+name]=args.posts/('transfer-'+('candidate' if name=='two_pixels' else name))
    collect(args.runs,['control','placed','one_pixel','two_pixels','combined','production'],
            args.out,'control','bec7c94',folders=paths)
    with frozen.archive(frozen.ARCHIVE) as z:
        retired=[r['relative_geometry'] for r in json.loads(z.read('regression/cases.json'))
                 if r['rgb_sha256'] in TEXTURE_ONLY_RGB]
    assert len(retired)==8 and all(r['placed']==0 for r in retired)
    fixed=read(primary/'summary.json')['groups']['fixed']
    target=sum(r['target'] for r in retired)
    historical_f1=2*fixed['correct']/(fixed['placed']+fixed['target']+target)
    save(args.out/'verification.json',dict(equivalent_exports=verified,cli=commands,
        retired_texture_controls=dict(frames=8,visible_targets=target,baseline_placements=0),
        historical_112_f1=historical_f1,primary_cases_sha256=digest(primary/'cases.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--control',type=Path,default=Path('../wall-section-round/runs/control'))
    parser.add_argument('--posts',type=Path,default=Path('../narrow-post-count/runs'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/visible-error-budget-01'))
    run(parser.parse_args())
