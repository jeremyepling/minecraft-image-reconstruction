"""Compare both final placement orders, the ordinary placer and two complete screenshot CLI builds."""
import argparse
from pathlib import Path

from benchmark.mask_blocks import decoded_pack
from benchmark.roof_span_verify import records, verify
from benchmark.texture_camera_eval import checked, digest, read, save


def compare(first,second):
    before,after=records(first),records(second)
    assert before.keys()==after.keys()
    for key,left in before.items():
        right=after[key]
        for field in ('rgb_sha256','reference_hashes','relative_geometry'):
            assert left[field]==right[field],(key,field)
        if 'input_hashes' in left:
            assert left['input_hashes']==right['input_hashes']
        split,frame=key
        a=checked(first/split/frame/'reconstruction.zip',left['pack_sha256'])
        b=checked(second/split/frame/'reconstruction.zip',right['pack_sha256'])
        assert decoded_pack(a)==decoded_pack(b),key
    return len(before)


def run(runs,out):
    winner='roof_then_walls'
    verify(runs,out,winner)
    report=read(out)
    equal_orders=sum(compare(runs/(prefix+winner),runs/(prefix+'walls_then_roof'))
                     for prefix in ('','transfer-','native-','side-','elevated-'))
    panels={prefix:compare(runs/(prefix+'-'+winner),runs/(prefix+'-production'))
            for prefix in ('native','side','elevated')}
    assert equal_orders==226 and sum(panels.values())==22
    extra=runs/'cli-rectangle'
    key='shape-benchmark-26.3--train-canopy-09-right'
    manifest=read(extra/'build-manifest.json')
    row=records(runs/'production')['old',key]
    placement=manifest['placement']
    cli=decoded_pack(extra/'reconstruction.zip')
    assert row['rgb_sha256']==manifest['image_sha256']
    assert cli==decoded_pack(runs/'production/old'/key/'reconstruction.zip')
    assert len(cli)==placement['placed_blocks']==placement['strict_blocks']+placement['added_blocks']
    assert placement['rectangle_roof_blocks']>0
    report.update(identical_orders=equal_orders,new_panel_reproductions=panels,
                  cli_rectangle=dict(id=key,rectangle_roof_blocks=placement['rectangle_roof_blocks'],
                                     total=len(cli),pack_sha256=digest(extra/'reconstruction.zip')))
    save(out,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.runs,args.out)
