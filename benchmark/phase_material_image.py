"""Read block IDs at measured spacing from an ordinary single-image Surface Finder run."""
import argparse
from pathlib import Path

from PIL import Image

from benchmark.camera_round import digest, read, save
from benchmark.known_spacing_atlas import predict
from benchmark.phase_material_match import create_matcher, predict_ids
from engine.atlas_count import load_atlas
from engine.masked_count import count
from engine.surface_sample import sample_surface
from runtime.mode import image_only


@image_only()
def run(image_path, surface_run, output, method='shape', atlas=None):
    build = read(surface_run/'build-manifest.json')
    image_sha = digest(image_path)
    if (build['image_sha256']!=image_sha or build['version']!='26.3'
            or build['surface_finder']['method']!='Regions First'):
        raise ValueError('Expected a 26.3 Surface Finder run for this exact screenshot')
    surface_path = surface_run/'surfaces.json'
    surfaces = read(surface_path)
    with Image.open(image_path) as src:
        image = src.convert('RGB')
    if list(image.size)!=surfaces['image_size']:
        raise ValueError('Different surface coordinate system')
    parts = surfaces['planes']
    model = create_matcher(method,atlas)
    samples = [sample_surface(image,p) for p in parts]
    ordinary = [count(s) for s in samples]
    locator_atlas = load_atlas(atlas) if atlas else None
    readings = [predict(s,b,atlas=locator_atlas) for s,b in zip(samples,ordinary)]
    faces = predict_ids(image,parts,samples,readings,model)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'materials.json',dict(image_sha256=image_sha,surfaces_sha256=digest(surface_path),
         source_manifest_sha256=digest(surface_run/'build-manifest.json'),matcher=model.provenance,
         surfaces=[dict(id=p['id'],quad=s['quad'] if s else None,counts=r['counts'],ordinary_phase=b['phase'],
                        phase=r['phase']) for p,s,b,r in zip(parts,samples,ordinary,readings,strict=True)],
         faces=faces))
    print(f'{len(faces)} visible face crops; {sum(f["block_id"] is not None for f in faces)} named.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('surface_run',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--matcher',choices=('shape','atlas'),default='shape')
    parser.add_argument('--atlas',type=Path)
    args = parser.parse_args()
    run(args.image,args.surface_run,args.out,args.matcher,args.atlas)
