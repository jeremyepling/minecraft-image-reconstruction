"""Collect the paired roof experiments and their exported changes against the shared control."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save


def collect(runs,names,out,baseline_name='roof-control',control_revision='99655be',*,folders=None):
    def run_path(name):
        return (folders or {}).get(name,runs/name)
    out.mkdir(parents=True,exist_ok=True)
    baseline=read(run_path(baseline_name)/'summary.json')
    previous=read(checked(run_path(baseline_name)/'cases.json',baseline['cases_sha256']))
    index={(r['split'],r['id']):r for r in previous}
    assert len(index)==200
    summaries={}
    with ZipFile(out/'comparisons.zip','w',ZIP_DEFLATED) as z:
        for name in names:
            here=run_path(name)
            meta=read(here/'summary.json')
            prior_sources=('sources','calibrated_atlas','roof_probes_sha256')
            assert all(meta[k]==baseline[k] for k in prior_sources)
            rows=read(checked(here/'cases.json',meta['cases_sha256']))
            assert len(rows)==len(index) and {(r['split'],r['id']) for r in rows}==index.keys()
            for r in rows:
                s,key=r['split'],r['id']
                last=index[s,key]
                assert r['control_pack_sha256']==last['pack_sha256']
                assert all(r[k]==last[k] for k in ('rgb_sha256','input_hashes','reference_hashes'))
                if r['changed']:
                    folder=here/s/key
                    pack=checked(folder/'reconstruction.zip',r['pack_sha256'])
                    assert set(decoded_pack(pack))==set(map(tuple,read(folder/'cells.json')['local']))
                    for f in (pack,folder/'cells.json'):
                        z.write(f,f'{name}/{s}/{key}/{f.name}')
                else:
                    assert r['pack_sha256']==r['control_pack_sha256']
            for f in ('summary.json','cases.json'):
                z.write(here/f,f'{name}/{f}')
            other=run_path('transfer-'+name)
            recent=read(other/'summary.json')
            extra_rows=read(checked(other/'cases.json',recent['cases_sha256']))
            assert len(extra_rows)==4
            for r in extra_rows:
                pack=checked(other/r['id']/'reconstruction.zip',r['pack_sha256'])
                z.write(pack,f"{name}/transfer/{r['id']}/reconstruction.zip")
            for f in ('summary.json','cases.json'):
                z.write(other/f,f'{name}/transfer/{f}')
            summaries[name]=dict(method=meta['candidate'],revision=meta['revision'],groups=meta['groups'],
                                 camera_registered=meta['camera_registered'],canopies=meta['canopies'],
                                 transfer=recent['relative_geometry'],cases_sha256=meta['cases_sha256'])
    with ZipFile(out/'comparisons.zip') as z:
        assert z.testzip() is None
    save(out/'results.json',dict(control=control_revision,archive_sha256=digest(out/'comparisons.zip'),methods=summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--baseline',default='roof-control')
    p.add_argument('--control-revision',default='99655be')
    p.add_argument('methods',nargs='+')
    a=p.parse_args()
    collect(a.runs,a.methods,a.out,a.baseline,a.control_revision)
