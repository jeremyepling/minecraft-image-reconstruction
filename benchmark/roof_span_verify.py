"""Check the promoted entry point against measured candidates and a complete CLI build."""
import argparse
from pathlib import Path

from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save


def records(folder):
    summary=read(folder/'summary.json')
    rows=read(checked(folder/'cases.json',summary['cases_sha256']))
    return {(r.get('split',''),r['id']):r for r in rows}


def verify(runs,out):
    checked_views=[]
    for prefix in ('','transfer-'):
        before,after=(runs/(prefix+name) for name in ('conservative','production'))
        candidate,produced=records(before),records(after)
        assert candidate.keys()==produced.keys()
        for (split,key),base in candidate.items():
            row=produced[split,key]
            assert row['rgb_sha256']==base['rgb_sha256'] and row['relative_geometry']==base['relative_geometry']
            a=checked(before/split/key/'reconstruction.zip',base['pack_sha256'])
            b=checked(after/split/key/'reconstruction.zip',row['pack_sha256'])
            assert decoded_pack(a)==decoded_pack(b)
        checked_views.append(len(candidate))
    key='shape-benchmark-26.3--train-canopy-06-left'
    src=runs/'cli-canopy'
    m=read(src/'build-manifest.json')
    p=next(v for v in m.values() if isinstance(v,dict) and 'strict_blocks' in v)
    cli=decoded_pack(src/'reconstruction.zip')
    source=records(runs/'production')['old',key]
    assert source['rgb_sha256']==m['image_sha256']
    assert cli==decoded_pack(runs/'production/old'/key/'reconstruction.zip')
    assert len(cli)==p['placed_blocks']==p['strict_blocks']+p['added_blocks']
    assert checked_views==[200,4]
    save(out,dict(decoded_exports_with_identical_coordinates_and_materials=sum(checked_views),
                  cases_sha256=read(runs/'production/summary.json')['cases_sha256'],
                  cli=dict(id=key,rgb_sha256=m['image_sha256'],pack_sha256=digest(src/'reconstruction.zip'),
                           total=len(cli),roof_span_blocks=p['roof_span_blocks'],
                           coordinates_and_materials_match=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    verify(a.runs,a.out)
