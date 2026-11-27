"""Verify the raw-image CLI export and replay the same model fields without its three-face hint."""
from pathlib import Path

from benchmark import framed_depth_eval as f
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import digest,read,save
from engine import atlas_plane_depth
from engine.camera_metric import predict


def main():
    root=Path('runs/rgb-handoff-recess')
    manifest=read(root/'build-manifest.json')
    split,fr,prior=next(j for j in f.cases(['old']) if j[1]['files']['rgb.png']==manifest['image_sha256'])
    image,_,_,_=f.photo(split,fr,prior)
    parts=read(root/'surfaces.json')['planes']
    counts=read(root/'counts.json')
    no_hint=[{k:v for k,v in p.items() if k!='inset_back_plane'} for p in parts]
    camera=predict(image,'26.3')
    placed=decoded_pack(root/'reconstruction.zip')
    ordinary,_=atlas_plane_depth.place_ranked(image,'26.3',no_hint,counts,camera)
    hinted,status=atlas_plane_depth.place_ranked(image,'26.3',parts,counts,camera)
    assert hinted==set(placed)
    truth,occupied,T,_=f.references(split,fr,f.NATIVE_INPUTS[split],f.TRAINING_DATA,f.DEPTH,
        dict(native=f.native.BUILDS,old=f.transfer.BUILDS,fresh=f.transfer.BUILDS))
    result=placement_scores(registered(hinted,T),registered(ordinary,T),dict.fromkeys(truth),occupied)
    row=next(r for r in read(Path('benchmark/reviews/inset-depth-handoff-01/comparison.json'))['cases'] if r['id']==fr['id'])
    m=result['relative_geometry']
    m.pop('joint_correct',None)
    assert m==row['geometry']
    declared={p['id']:p['inset_back_plane'] for p in parts if p.get('inset_back_plane')}
    chosen={r['id']:r['level'] for r in status['plane_offsets'] if r['id'] in declared}
    save(Path('benchmark/reviews/inset-depth-handoff-01/rgb-cli.json'),dict(id=fr['id'],rgb=manifest['image_sha256'],
         model_sha256=manifest['model_sha256'],artifacts={name:digest(root/name) for name in
         ('surfaces.json','counts.json','placements.json','reconstruction.zip')},measured=declared,chosen=chosen,
         placed=len(placed),source_only_matches_full_replay=True,placement_delta=m))


if __name__=='__main__':
    main()
