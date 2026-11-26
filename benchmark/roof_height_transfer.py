"""Score the height rule on the four framed shelter and gate photos against 54c9fee."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
from unittest.mock import patch

from benchmark import visible_depth_transfer as transfer, visible_depth_score as scorer
from benchmark.mask_blocks import decoded_pack
from benchmark.roof_height_eval import CHANGES, CONTROL, changes, previous_rows
from benchmark.texture_camera_eval import ROOT, checked, digest, read, save
from benchmark.visible_recovery_report import summarize


def run(inputs,root,method,out,*,bundle=None,prefix='conservative/transfer',control=CONTROL,archive_path=CHANGES):
    review = ROOT/'benchmark/reviews/framed-depth-transfer-01'
    screen = checked(inputs/'screen.json',digest(review/'screen.json'))
    frames = read(screen)['selected']
    if bundle is None:
        bundle=changes()
    previous = {r['id']:r for r in previous_rows(prefix,bundle)}
    assert len(previous)==len(frames)==4 and set(previous)=={f['id'] for f in frames}
    framing = read(review/'framing.json')
    assert all(r['framing']['accepted'] for r in framing)
    assert {r['id']:r['files'] for r in framing}=={f['id']:f['files'] for f in frames}
    scorer.init_candidate(root,method)
    callback = scorer.predict
    rows = []
    for f in frames:
        key = f['id']
        prior = previous[key]
        assert prior['rgb_sha256']==f['files']['rgb.png']
        for name,field in [('surfaces.json','surface_sha256'),('counts.json','counts_sha256')]:
            checked(inputs/'upstream'/key/name,prior[field])
        pack = bundle.read(f'{prefix}/{key}/reconstruction.zip')
        assert sha256(pack).hexdigest()==prior['pack_sha256']
        before = set(decoded_pack(BytesIO(pack)))
        status = {}
        def predict(image,version,parts,counts,camera):
            result = callback(image,parts,counts,camera,before.copy()) if callback else before
            if isinstance(result,tuple):
                cells, details = result
                status.update(details)
                return cells,details
            return result,{}
        with patch.object(scorer,'predict',predict), patch.object(transfer,'BUILDS',inputs):
            row = transfer.score_case((f,inputs,'comparison',out,None),baseline_local=before)
        for field in ('relative_geometry','metrics'):
            m = row[field]
            assert m['correct']-m['gained']+m['lost']==prior[field]['correct']
            assert m['extras']-m['extra_added']+m['extra_removed']==prior[field]['extras']
            assert m['hidden_extras']-m['hidden_added']+m['hidden_removed']==prior[field]['hidden_extras']
        assert list(row['baseline_translation'])==prior['translation']
        row['relative_geometry'].pop('joint_correct')
        row.update(status=status,control_pack_sha256=prior['pack_sha256'],reference_hashes=f['files'])
        rows.append(row)
    save(out/'cases.json',rows)
    summary=dict(method=method,control=control,revision=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=root,text=True).strip(),screen_sha256=digest(screen),
        control_archive_sha256=digest(archive_path),framing_sha256=digest(review/'framing.json'),
        cases_sha256=digest(out/'cases.json'),relative_geometry=summarize(rows))
    if method:
        module = root.joinpath(*method.partition(':')[0].split('.')).with_suffix('.py')
        summary['adapter_sha256'] = digest(module)
    save(out/'summary.json',summary)
    print(summary['relative_geometry'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--candidate-root',type=Path,default=ROOT)
    parser.add_argument('--candidate')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.inputs.resolve(),args.candidate_root.resolve(),args.candidate,args.out)
