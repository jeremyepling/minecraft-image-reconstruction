"""Reserve the unused RGB views of previously evaluated native building geometries."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark import canopy_fixture, visible_depth_transfer as transfer
from benchmark.texture_camera_eval import STABLE, checked, digest, read, save

DEST = Path(__file__).resolve().parent/'reviews/visible-recovery-extra-views-01'
COHORTS = ('validation', 'test', 'regression', 'native', 'old', 'fresh')


def reserve():
    source = transfer.OUT/'screen.json'
    screen = read(checked(source, canopy_fixture.SCREEN_SHA))
    prior = STABLE/'runs/canopy-placement/release'
    excluded = set()
    previous_shapes = set()
    manifests = {}
    for split in COHORTS:
        cases = prior/split/'cases.json'
        rows = read(cases)
        excluded.update(row['rgb_sha256'] for row in rows)
        manifests[split] = dict(sha256=digest(cases), frames=len(rows))
    for file in (source, canopy_fixture.OUT/'screen.json'):
        for row in read(file)['selected']:
            assert row['files']['rgb.png'] in excluded
            previous_shapes.add(row['shape'])
    supported = [f for f in screen['screened'] if f['camera']['P'] is not None]
    selected = {}
    for f in sorted(supported, key=lambda r:r['files']['rgb.png']):
        sha = f['files']['rgb.png']
        if sha not in excluded:
            selected.setdefault(sha, f)
    frames = list(selected.values())
    shapes = {f['shape'] for f in frames}
    result = dict(selection='every previously unevaluated distinct RGB with a supported frozen camera',
                  source_sha256=digest(source), confirmation_exclusion_sha256=digest(canopy_fixture.OUT/'screen.json'),
                  prior_case_manifests=manifests, screened=len(screen['screened']), supported=len(supported),
                  supported_distinct_rgb=len({f['files']['rgb.png'] for f in supported}),
                  excluded_fixture_images=len(excluded), distinct_camera_supported_excluded=len({f['files']['rgb.png'] for f in supported}&excluded),
                  distinct_new_views=len(frames), distinct_shapes=len(shapes), previously_evaluated_shapes=len(shapes&previous_shapes),
                  selected=frames)
    DEST.mkdir(parents=True,exist_ok=True)
    save(DEST/'screen.json',result)
    print({k:v for k,v in result.items() if k not in ('selected','prior_case_manifests')})
    print([f['id'] for f in frames])



def package(run, archive):
    """Pack the original frames and exactly the frozen prediction/scoring inputs."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    rows = read(DEST/'screen.json')['selected']
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
        zipped.write(DEST/'screen.json', 'confirmation/screen.json')
        for row in rows:
            png = transfer.rgb_path(row)
            for path, label in ((png, 'rgb.png'), (png.with_suffix('.json'), 'camera.json'),
                                (png.with_name(row['source_id']+'-blocks.json'), 'world-readback.json')):
                checked(path, row['files'][label])
                zipped.write(path, 'fixtures/'+path.relative_to(transfer.BUILDS).as_posix())
            upstream = run/'upstream'/row['id']
            manifest = read(upstream/'build-manifest.json')
            if manifest['image_sha256'] != row['files']['rgb.png'] or manifest['through'] != 'counts':
                raise ValueError(f"Upstream image mismatch for {row['id']}")
            for name in ('build-manifest.json', 'surfaces.json', 'counts.json'):
                zipped.write(upstream/name, f"confirmation/upstream/{row['id']}/{name}")
    return dict(bytes=archive.stat().st_size, sha256=digest(archive))


def roof_checks(run):
    from benchmark import relative_canopy_spacing as roof
    specs = read(roof.PROBES)
    examples = {f['id']:f for p in (transfer.OUT, canopy_fixture.OUT) for f in read(p/'screen.json')['selected']}
    rows = []
    for f in read(DEST/'screen.json')['selected']:
        equivalent = [k for k in specs if k.rsplit('-',1)[0] == f['id'].rsplit('-',1)[0]]
        if not equivalent:
            continue
        def occupied(frame):
            path = transfer.rgb_path(frame).with_name(frame['source_id']+'-blocks.json')
            return transfer.structure(read(checked(path,frame['files']['world-readback.json'])),64)
        original = examples[equivalent[0]]
        source = occupied(f)
        if source != occupied(original):
            raise ValueError('Matched canopy views have different world-coordinate geometry')
        spec = specs[original['id']]
        pose = read(checked(transfer.rgb_path(f).with_suffix('.json'), f['files']['camera.json']))
        faces,_ = roof.raster(pose, source)
        visible = {tuple(face['xyz_world']) for face in faces}
        row = dict(id=f['id'], previous_probe=original['id'])
        for method in ('previous', 'candidate'):
            cells, pack_sha = roof.verify_pack(f, run/method, pose)
            row[method] = dict(roof.score(cells,visible,spec), pack_sha256=pack_sha)
        rows.append(row)
    return rows


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('reserve','package','roofs'), default='reserve', nargs='?')
    parser.add_argument('--run', type=Path, default=Path('runs/extra-views'))
    parser.add_argument('--archive', type=Path, default=Path('runs/extra-views-v1.zip'))
    args = parser.parse_args()
    if args.stage == 'reserve':
        reserve()
    elif args.stage == 'package':
        print(package(args.run, args.archive))
    else:
        save(DEST/'roofs.json', roof_checks(args.run))
