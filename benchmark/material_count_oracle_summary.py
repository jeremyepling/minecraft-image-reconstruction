"""Collect paired ordinary pack results and bind the published evidence to its decoded coordinates."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark import material_feedback_grid as trial
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import digest, read, save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folders',type=Path,nargs='+')
    args=ap.parse_args()
    rows=[]
    known=set()
    modes=('control','sham','feedback','oracle','disabled')
    bundle=trial.OUT/'count-only-exports.zip'
    with ZipFile(bundle,'w',ZIP_DEFLATED) as archive:
        for folder in sorted(args.folders):
            r=read(folder/'result.json')
            if r['id'] in known or not r['matched_release']:
                raise ValueError('Non-paired or duplicate source')
            known.add(r['id'])
            control=decoded_pack(folder/'control/reconstruction.zip')
            site=trial.typed_inputs()[r['id']]
            if control!={tuple(v[:3]):v[3] for v in site['rows']}:
                raise ValueError('Control pack diverges from published typed release')
            camera=read(folder/'control/build-manifest.json')['placement']['P']
            if any(read(folder/m/'build-manifest.json')['placement']['P']!=camera for m in modes):
                raise ValueError('Estimated camera changed between paired exports')
            for file,k in [('surfaces.json','masks_sha256'),('counts.json','counts_sha256')]:
                r[k]=digest(folder/'control'/file)
                if any(digest(folder/m/file)!=r[k] for m in modes):
                    raise ValueError('Upstream masks or counts changed')
            for m in modes:
                path=folder/m/'reconstruction.zip'
                data=path.read_bytes()
                pred=decoded_pack(BytesIO(data))
                row=r['cases'][m]
                if digest(path)!=row['pack_sha256']:
                    raise ValueError('Pack digest mismatch')
                archive.writestr(r['id']+'/'+m+'.zip',data)
                row['local_coordinates_added']=len(pred.keys()-control.keys())
                row['local_coordinates_removed']=len(control.keys()-pred.keys())
                row['local_ids_changed']=sum(pred[p]!=control[p] for p in pred.keys()&control.keys())
                row['id_sources']=dict(Counter(b['material_source'] for b in read(folder/m/'placements.json')['blocks']))
                status=read(folder/m/'build-manifest.json')['placement']
                row['ranked_strict_blocks']=status['strict_blocks']
                row['ranked_added_blocks']=status['added_blocks']
                # Evidence comes from the emitted JSON and IDs from the templates.
                row['typed']={k:row['typed'][k] for k in ('strict','facade','material')}
            rows.append(r)
    summary={}
    for m in modes:
        metrics={}
        for stage in ('geometry','typed'):
            for metric in ('strict','facade'):
                counts=[sum(r['cases'][m][stage][metric][k] for r in rows) for k in ('correct','missed','extras')]
                metrics[stage+'_'+metric]=trial.typed.overlap(*counts)
        summary[m]=dict(metrics=metrics,count_edits=sum(len(r['cases'][m]['changes']) for r in rows),
                        coords_added=sum(r['cases'][m]['local_coordinates_added'] for r in rows),
                        coords_removed=sum(r['cases'][m]['local_coordinates_removed'] for r in rows),
                        ids_changed=sum(r['cases'][m]['local_ids_changed'] for r in rows))
        if m=='disabled':
            changed=[c for r in rows for c in r['cases'][m]['changes']]
            summary[m]['complete_spacings_disabled']=sum(None not in c[1] for c in changed)
            summary[m]['partial_spacings_disabled']=sum(None in c[1] for c in changed)
    save(trial.OUT/'count-only-oracle.json',dict(archive_sha256=digest(bundle),summary=summary,frames=rows))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
