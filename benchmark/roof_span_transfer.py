"""Replay a roof span method against the four previously reserved shelter and gate images."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from benchmark import visible_depth_transfer as transfer, visible_depth_score as scorer
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import ROOT, checked, digest, read, save
from benchmark.visible_recovery_report import summarize


def run(inputs, root, method, out):
    review=ROOT/'benchmark/reviews/framed-depth-transfer-01'
    info=read(review/'summary.json')
    screen=checked(inputs/'screen.json',digest(review/'screen.json'))
    frames=read(screen)['selected']
    scorer.init_candidate(root,method)
    callback=scorer.predict
    rows=[]
    with ZipFile(checked(review/'scores.zip',info['scores_sha256'])) as z, TemporaryDirectory() as temp:
        records=z.read('final/cases.json')
        assert sha256(records).hexdigest()==info['methods']['final']['cases_sha256']
        previous={r['id']:r for r in json.loads(records)}
        assert set(previous)=={f['id'] for f in frames}
        for f in frames:
            key=f['id']
            old=previous[key]
            for name,field in [('surfaces.json','surface_sha256'),('counts.json','counts_sha256')]:
                checked(inputs/'upstream'/key/name,old[field])
            assert old['rgb_sha256']==f['files']['rgb.png']
            pack=z.read(f'final/{key}/reconstruction.zip')
            assert sha256(pack).hexdigest()==old['pack_sha256']
            before=set(decoded_pack(BytesIO(pack)))
            coords=z.read(f'final/{key}/cells.json')
            assert before==set(map(tuple,json.loads(coords)['local']))
            baseline=Path(temp)/key
            baseline.mkdir()
            (baseline/'cells.json').write_bytes(coords)
            def predict(image,version,parts,counts,camera):
                result=callback(image,parts,counts,camera,set(before)) if callback else before
                return result if isinstance(result,tuple) else (result,{})
            with patch.object(scorer,'predict',predict), patch.object(transfer,'BUILDS',inputs):
                row=transfer.score_case((f,inputs,'comparison',out,Path(temp)))
            m=row['relative_geometry']
            assert m['correct']-m['gained']+m['lost']==old['relative_geometry']['correct']
            m.pop('joint_correct',None)
            rows.append(row)
    save(out/'cases.json',rows)
    summary=dict(method=method,control='99655be',revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=root,text=True).strip(),screen_sha256=digest(screen),
        cases_sha256=digest(out/'cases.json'),relative_geometry=summarize(rows))
    save(out/'summary.json',summary)
    print(summary['relative_geometry'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,default=ROOT)
    p.add_argument('--candidate')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run(args.inputs.resolve(),args.candidate_root.resolve(),args.candidate,args.out)
