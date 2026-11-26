"""Reserve the unused RGB views of previously evaluated native building geometries."""
from pathlib import Path

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


if __name__=='__main__':
    reserve()
