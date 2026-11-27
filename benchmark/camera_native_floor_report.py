"""Aggregate the frozen research composition and its model-free production subset."""
from collections import Counter
from importlib.metadata import version
import platform
from pathlib import Path

from benchmark.camera_model_controls import inventory_hash, summary
from benchmark.camera_native_floor_audit import SCORER
from benchmark.camera_native_floor_run import CONTROL, REVIEW, ROOT, RUN, read, save, digest
from benchmark.camera_geocalib_infer import WEIGHT_SHA
from engine.camera_atlas_floor import _atlas
from benchmark.mask_blocks import decoded_pack

FREEZE = 'd6990f9d0a02b53c26746155580b7a019cd18865'


def main():
    baseline = read(CONTROL/'independent-matched-cases.json.gz')
    frozen = read(CONTROL/'frames.json')
    inv = {r['id']:r for r in read(CONTROL/'inventory.json.gz')}
    outcomes = {r['id']:r for r in read(REVIEW/'native-cameras.json.gz')}
    decoded = {}
    arms = {'boundary': [], 'composed': []}
    for old in baseline:
        key=old['id']
        result=outcomes.get(key)
        assert (result is None) == old['camera_usable']
        row=old
        if result and result.get('relative'):
            row=read(RUN/'scores'/(key+'.json'))
            folder=RUN/'builds'/old['dataset']/key
            manifest=read(folder/'build-manifest.json')
            pack=folder/'reconstruction.zip'
            assert row['source_sha256']==result['rgb_sha256']==inv[key]['rgb_sha256']==manifest['image_sha256']
            assert row['reference_sha256']==inv[key]['source_reference_sha256']
            assert manifest['placement']['P']==result['camera']['P']
            assert row['pack_sha256']==digest(pack)
            cells=decoded_pack(pack)
            assert len(cells)==row['exported']
            decoded[key]=dict(source_sha256=row['source_sha256'], pack_sha256=row['pack_sha256'],
                              decoded_sha256=inventory_hash(cells), exported=len(cells), arm=result['choice'])
            row=dict(row, provenance='new native floor camera, decoded ordinary export',
                     decoded_sha256=decoded[key]['decoded_sha256'])
        arms['composed'].append(row)
        arms['boundary'].append(row if result and result['choice']=='boundary' else old)
    save(REVIEW/'build-receipts.json.gz',decoded)
    scores={name:summary(rows,frozen) for name,rows in arms.items()}
    scores['incumbent']=read(CONTROL/'independent-matched-summary.json')['datasets']
    save(REVIEW/'summary.json',scores)
    save(REVIEW/'cases.json.gz',arms['composed'])
    save(REVIEW/'production-cases.json.gz',arms['boundary'])
    manifest=dict(source_freeze=FREEZE,engine_revision='aac7da7',base='62611ee',
                  native_corner_scorer_sha256=digest(SCORER),source_atlas=_atlas('26.3').provenance,geocalib_weights_sha256=WEIGHT_SHA,
                  environment=dict(python=platform.python_version(),numpy=version('numpy'),opencv=__import__('cv2').__version__),dataset='Java 26.3 vanilla, saved screenshots',
                  accepted_by_split=dict(Counter(r['split'] for r in outcomes.values() if 'relative' in r)),
                  retained_supported_exports=sum(r['camera_usable'] for r in baseline),
                  scoring='one whole-build translation; ordinary facade exemptions and material tie rules',
                  source_policy='boundary-native then optional RGB GeoCalib proposal; production is boundary only',
                  inputs={f:digest(CONTROL/f) for f in ('frames.json','inventory.json.gz','manifest.json',
                                                            'independent-matched-cases.json.gz','independent-matched-summary.json')},
                  outputs={f.relative_to(REVIEW).as_posix():digest(f) for f in REVIEW.rglob('*') if f.is_file() and f.name not in ('README.md','manifest.json')})
    save(REVIEW/'manifest.json',manifest)


if __name__=='__main__':
    main()
