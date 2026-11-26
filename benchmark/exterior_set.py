"""Load a versioned exterior fixture release. All stages share the same case IDs."""
from dataclasses import dataclass
from pathlib import Path

from benchmark.camera_round import ROOT, digest, fixture_digest, read, validate


RELEASE = ROOT / 'benchmark/suites/exterior-v1.json'


@dataclass(frozen=True)
class Capture:
    floor: str
    root: Path
    frame: dict

    def path(self, key):
        return self.root / self.frame[key]

    @property
    def id(self):
        return self.frame['id']


def pair_cases(by_floor):
    """Pair by layout/palette/pose identity; never zip indexes by file order."""
    groups = {}
    identity = ('layout', 'split', 'palette', 'angle', 'condition')
    for floor, cases in by_floor.items():
        for case in cases:
            group = groups.setdefault(case.id, {})
            if floor in group:
                raise ValueError(f'Duplicate frame {floor}/{case.id}')
            if group:
                first = next(iter(group.values()))
                if any(first.frame[k] != case.frame[k] for k in identity):
                    raise ValueError(f'Mismatched identity for {case.id}')
            group[floor] = case
    for name, group in groups.items():
        if group.keys() != by_floor.keys():
            raise ValueError(f'Missing floor variant for {name}')
    return dict(sorted(groups.items()))


def check_capture_conditions(pose, expected, image_id):
    for key, value in expected.items():
        if key not in pose or pose[key] != value:
            raise ValueError(f'{image_id}: recorded {key}={pose.get(key)!r}, expected {value!r}')


def load_suite(path=RELEASE):
    suite = read(path)
    if suite['schema'] != 1:
        raise ValueError('Unsupported exterior suite schema')
    floors = {}
    for floor, spec in suite['conditions'].items():
        root = (path.parent / spec['index']).resolve().parent
        index, splits = validate(root)
        if (index.get('floor', 'smooth_stone') != floor or index['version'] != suite['version']
                or index['client_sha256'] != suite['client_sha256'] or splits != suite['layouts']
                or len(index['frames']) != suite['pairs']):
            raise ValueError(f'Fixture membership or capture version differs for {floor}')
        for frame in index['frames']:
            pose = read(root / frame['pose'])
            check_capture_conditions(pose, suite['capture_conditions'], f"{floor}/{frame['id']}")
        if fixture_digest(root, index) != spec['sha256']:
            raise ValueError(f'Frozen fixture content changed for {floor}')
        floors[floor] = [Capture(floor, root, f) for f in index['frames']]
    audit = suite['pair_audit']
    if digest(path.parent / audit['path']) != audit['sha256']:
        raise ValueError('Frozen floor pairing audit changed')
    worlds = suite['world_archives']
    world_index = path.parent / worlds['index']
    if digest(world_index) != worlds['sha256']:
        raise ValueError('Frozen world index changed')
    saves = read(world_index)['worlds']
    if {s['floor'] for s in saves} != set(floors):
        raise ValueError('World floor conditions differ from the release')
    for save in saves:
        check_capture_conditions(save, dict(clock_ticks=6000, raining=False, thundering=False,
                                           advance_time=False, advance_weather=False), save['archive'])
        if digest(world_index.parent / save['archive']) != save['sha256']:
            raise ValueError(f"Frozen world archive changed: {save['archive']}")
    return suite, pair_cases(floors)
