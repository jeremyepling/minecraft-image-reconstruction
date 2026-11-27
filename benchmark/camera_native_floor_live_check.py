"""Verify real normal CLI builds on pinned local vision models and the usual Windows launcher."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

from benchmark.camera_model_controls import inventory_hash
from benchmark.camera_native_floor_run import CONTROL, REVIEW, RUN, digest, read, save
from benchmark.expanded_fixed_handoffs import _release, fixed_frames, source_inputs
from benchmark.mask_blocks import decoded_pack


def main():
    baseline={r['id']:r for r in read(CONTROL/'independent-matched-cases.json.gz')}
    seam='corner-mixed-26.3--heldout-facade-oblique-left'
    recovered='validation-harbor-02-seafoam-rear-oblique'
    fixed=next(e for e in fixed_frames() if e['id']==seam)
    _,_,archive=_release()
    with ZipFile(archive) as reader:
        zipped=reader.read(seam+'/reconstruction.zip')
    assert sha256(zipped).hexdigest()==baseline[seam]['pack_sha256']
    checks=[(seam,RUN/'normal-cli'/seam,source_inputs(fixed)[3]['P'],decoded_pack(BytesIO(zipped)))]
    checks.append((recovered,RUN/'live-primary'/recovered,read(RUN/'boundary/metric'/(recovered+'.json'))['camera']['P'],
                   decoded_pack(RUN/'builds/visible-builds-v3'/recovered/'reconstruction.zip')))
    records=[]
    for key,folder,projection,cells in checks:
        manifest=read(folder/'build-manifest.json')
        placement=manifest['placement']
        assert baseline[key]['source_sha256']==manifest['image_sha256']
        assert placement['P']==projection
        assert decoded_pack(folder/'reconstruction.zip')==cells
        records.append(dict(id=key,camera=placement['camera_source'],source_sha256=manifest['image_sha256'],
                            exact_matrix=True,verified_decoded_sha256=inventory_hash(cells),placed=len(cells),
                            cli_pack_sha256=digest(folder/'reconstruction.zip'),
                            native_floor_evidence=placement.get('native_floor_evidence')))
    save(REVIEW/'live-primary-cli.json',dict(launcher='Invoke-WallRecovery.ps1, default primary Python, live pinned local model fields',cases=records))


if __name__=='__main__':main()
