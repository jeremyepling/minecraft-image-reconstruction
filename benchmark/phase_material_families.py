"""Load the eight prespecified Visible Builds frames, their image-only outlines and deferred grading truth."""
import argparse
from functools import partial
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.seam_family_check import RELEASE
from benchmark.surface_metrics import truth


SPLITS = {'twinspire1': 'test', 'cloister1': 'validation'}
REVIEW = ROOT/'benchmark/reviews/atlas-phase-materials-01'
PRIOR = ROOT/'benchmark/reviews/block-seam-01'
OUTPUTS = ('surfaces.json', 'build-manifest.json')


def roster(layout):
    release = read(RELEASE)
    split = SPLITS[layout]
    frames = {r['id']: r['rgb_sha256'] for r in release['frames'] if r['layout']==layout and r['split']==split}
    expected = {f'{split}-{layout}-{palette}-{angle}' for palette in ('cool', 'earth')
                for angle in ('high-right', 'low-left')}
    if set(frames)!=expected or release['minecraft_version']!='26.3':
        raise ValueError(f'Changed Visible Builds v1 frame roster for {layout}')
    return frames, release['shards'][split]['sha256']


def checked_member(archive, name, expected):
    data = archive.read(name)
    if sha256(data).hexdigest()!=expected:
        raise ValueError('Checksum mismatch on '+name)
    return data


def load_reference(shard_path, prefix, hashes):
    # Open the grading files only when the caller has finished producing material predictions.
    with ZipFile(shard_path) as archive:
        pose = json.loads(checked_member(archive, prefix+'camera.json', hashes['camera.json']))
        faces = json.loads(checked_member(archive, prefix+'reference-faces.json', hashes['reference-faces.json']))['faces']
    return truth(pose, faces), faces


def iter_family(layout, shard_path, surfaces_zip):
    """Yield frame ID, original RGB hash, RGB image, image-only proposals, and load_reference()."""
    frames, shard_sha = roster(layout)
    if digest(Path(shard_path))!=shard_sha:
        raise ValueError('Unexpected dataset shard for '+layout)
    if layout=='cloister1':
        manifest = read(REVIEW/'cloister-manifest.json')
        expected_zip = manifest['surface_archive_sha256']
        records = manifest['frames']
    else:
        expected_zip = read(PRIOR/'manifest.json')['raw_family_surfaces_sha256']
        records = {r['frame']: r for r in read(PRIOR/'family/cases.json')}
    if digest(Path(surfaces_zip))!=expected_zip:
        raise ValueError('Unexpected surface archive for '+layout)
    with ZipFile(shard_path) as images, ZipFile(surfaces_zip) as outputs:
        members = sorted(frame+'/'+name for frame in frames for name in OUTPUTS)
        if sorted(outputs.namelist())!=members or set(records)!=set(frames):
            raise ValueError('Unexpected surface archive inventory for '+layout)
        for frame, rgb_sha in sorted(frames.items()):
            prefix = 'frames/'+frame+'/'
            fr = json.loads(images.read(prefix+'frame.json'))
            if (fr['id']!=frame or fr['layout']!=layout or fr['split']!=SPLITS[layout] or
                    fr['version']!='26.3' or fr['files']['rgb.png']!=rgb_sha):
                raise ValueError('Unexpected capture identity on '+frame)
            rgb = checked_member(images,prefix+'rgb.png',rgb_sha)
            with Image.open(BytesIO(rgb)) as src:
                image = src.convert('RGB')
            rec = records[frame]
            build = json.loads(checked_member(outputs,frame+'/build-manifest.json',rec['build_manifest_sha256']))
            model = json.loads(checked_member(outputs,frame+'/surfaces.json',rec['surfaces_sha256']))
            if (build['image_sha256']!=rgb_sha or build['version']!='26.3' or build['through']!='surfaces'
                    or build['surface_finder']['method']!='Regions First' or model['image_size']!=list(image.size)):
                raise ValueError('Surface Finder provenance mismatch on '+frame)
            yield frame, rgb_sha, image, model['planes'], partial(load_reference, shard_path, prefix, fr['files'])


def package_cloister(source, destination):
    """Preserve the original region records and build manifests and verify their model identities."""
    frames, shard_sha = roster('cloister1')
    destination.mkdir(parents=True,exist_ok=True)
    archive_path = destination/'cloister-surfaces.zip'
    prior = read(PRIOR/'family/cases.json')[0]
    records = {}
    with ZipFile(archive_path,'x') as archive:
        for frame,rgb_sha in sorted(frames.items()):
            folder = source/frame
            build = read(folder/'build-manifest.json')
            if (build['image_sha256']!=rgb_sha or build['version']!='26.3' or build['through']!='surfaces'
                    or build['model_sha256']!=prior['models'] or build['stages']!=prior['stages']):
                raise ValueError('Expected the same ordinary image-only Surface Finder on '+frame)
            records[frame] = dict(rgb_sha256=rgb_sha,surfaces_sha256=digest(folder/'surfaces.json'),
                                  build_manifest_sha256=digest(folder/'build-manifest.json'))
            for name in OUTPUTS:
                archive.writestr(ZipInfo(frame+'/'+name),(folder/name).read_bytes(),compress_type=ZIP_DEFLATED)
    save(destination/'cloister-manifest.json',dict(layout='cloister1',split='validation',version='26.3',
         finder_source='d95c350',dataset_release='visible-builds-v1',shard_sha256=shard_sha,
         command='Invoke-WallRecovery.ps1 -Image <frame-dir>/rgb.png -Through surfaces -Out <out>/<frame-id>',
         model_sha256=prior['models'],stages=prior['stages'],frames=records,surface_archive_sha256=digest(archive_path)))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cloister_surface_directory',type=Path)
    parser.add_argument('output_directory',type=Path)
    args = parser.parse_args()
    package_cloister(args.cloister_surface_directory,args.output_directory)
