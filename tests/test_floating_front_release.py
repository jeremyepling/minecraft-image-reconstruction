import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from training import floating_front_release as front
from training import floating_shared_inputs as shared


class FloatingFrontReleaseTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.archives = self.root / 'releases'
        self.release = self.archives / front.DATASET
        self.fixture = self.root / 'fixture'
        self.pin = self.root / front.PIN.name
        self.enterContext(patch.object(front, 'PIN', self.pin))
        self.enterContext(patch.object(front, 'LAYOUTS', 2))
        self.enterContext(patch('training.front_structures.build_archive', self.build_structures))
        self.batches = [self.root / 'batch-a', self.root / 'batch-b']
        self.write_layout(self.batches[0], 'front-001-arcade', 'arcade', 'train')
        self.write_layout(self.batches[1], 'front-002-terrace', 'terrace', 'test')

    def build_structures(self, capture_roots, out):
        self.assertEqual(set(capture_roots), set(self.batches))
        with ZipFile(out, 'w') as archive:
            front.write_zip(archive, 'test-volume.nbt', b'test structure fixtures')
        return dict(file=out.name, sha256=front.sha256(out), bytes=out.stat().st_size, layouts=2, templates=4)

    def write_layout(self, batch, layout, family, split):
        for key in front.SHARDS:
            condition, lighting = key.split('-')
            marker = f'{layout}-{key}'
            frame = batch / 'frames' / marker
            frame.mkdir(parents=True)
            files = {
                'rgb.png': (f'{layout}-full-noon' if lighting == 'afternoon' else marker).encode(),
                'camera.json': front.encode(dict(id=marker, image_size=[900, 600], eye_world=[0, 0, 0],
                                                 clock_ticks=front.CLOCK[lighting],
                                                 view_rotation_projection=[1] * 16)),
                'labels.json': front.encode(dict(frame_id=marker, blocks=[condition])),
                'labels.npz': condition.encode(),
                'world-readback.json': front.encode([]),
                'reference-faces.json': front.encode({'extra': marker}),
            }
            for name, payload in files.items():
                (frame / name).write_bytes(payload)
            row = dict(id=marker, layout=layout, family=family, split=split, palette='stone', size='small',
                       angle='front', dimensions=[8, 8, 8], lighting=lighting, condition=condition,
                       parent_id=f'{layout}-full-noon', clock_ticks=front.CLOCK[lighting], ground_y=64,
                       framing={'accepted': True}, counts={'visible_blocks': 1},
                       files={name: front.sha256(frame / name) for name in files})
            (frame / 'frame.json').write_bytes(front.encode(row))

    def replace_asset(self, index, name, payload):
        frame = self.batches[0] / 'frames' / f'front-001-arcade-{index}'
        path = frame / name
        path.write_bytes(front.encode(payload))
        metadata = frame / 'frame.json'
        row = json.loads(metadata.read_bytes())
        row['files'][name] = front.sha256(path)
        metadata.write_bytes(front.encode(row))

    def stage(self):
        digest = front.package(self.batches, self.release)
        self.pin.write_text(digest, encoding='utf-8')
        self.assertEqual(shared.install(self.fixture, self.archives, front.DATASET), 8)
        return digest

    def test_package_is_deterministic_and_default_replay_uses_noon(self):
        digest = self.stage()
        second = self.root / 'second-release'
        self.assertEqual(front.package(reversed(self.batches), second), digest)
        self.assertEqual(shared.install(self.fixture, self.archives, front.DATASET), 8)
        staged = self.fixture / 'frames' / front.DATASET / front.STRUCTURES
        self.assertEqual(staged.read_bytes(), (self.release / front.STRUCTURES).read_bytes())
        default = shared.cases(self.fixture)
        self.assertEqual([row['id'] for row in default], ['front-001-arcade-full-noon', 'front-002-terrace-full-noon'])
        all_full = shared.cases(self.fixture, lighting='all')
        self.assertEqual(len(all_full), 6)
        self.assertEqual({row['split'] for row in all_full if row['layout'] == 'front-001-arcade'}, {'train'})
        selected = 'front-001-arcade-full-afternoon'
        afternoon, = shared.cases(self.fixture, lighting='afternoon', ids=[selected])
        self.assertEqual(afternoon['parent_id'], default[0]['id'])
        self.assertEqual(afternoon['dataset_manifest_sha256'], digest)
        self.assertEqual(afternoon['rgb_sha256'], default[0]['rgb_sha256'])
        self.assertNotEqual(afternoon['reference_sha256']['camera.json'], default[0]['reference_sha256']['camera.json'])
        self.assertEqual(afternoon['reference_sha256']['labels.npz'], default[0]['reference_sha256']['labels.npz'])
        decor = shared.cases(self.fixture, cohort='front-decorative')
        self.assertEqual([row['parent_id'] for row in decor], [row['id'] for row in default])
        self.assertEqual({row['lighting'] for row in decor}, {'noon'})
        self.assertFalse((Path(afternoon['frame']) / 'world-readback.json').exists())
        with ZipFile(self.release / front.shard_name('full-afternoon')) as archive:
            auxiliary = json.loads(archive.read(f'frames/{selected}/reference-faces.json'))
        self.assertEqual(auxiliary, {'extra': selected})
        with self.assertRaisesRegex(ValueError, 'Unknown front-full noon views'):
            shared.cases(self.fixture, ids=[selected])
        with self.assertRaisesRegex(ValueError, 'Unsupported lighting for front-decorative'):
            shared.cases(self.fixture, cohort='front-decorative', lighting='morning')

    def test_archives_and_staged_references_are_verified(self):
        digest = front.package(self.batches, self.release)
        with self.assertRaisesRegex(ValueError, 'manifest fingerprint'):
            shared.install(self.fixture, self.archives, front.DATASET, release_sha256='0' * 64)
        unsuitable = self.release / front.shard_name('full-afternoon')
        original = unsuitable.read_bytes()
        unsuitable.write_bytes(original + b'tamper')
        with self.assertRaisesRegex(ValueError, 'Archive fingerprint changed'):
            shared.install(self.fixture, self.archives, front.DATASET, release_sha256=digest)
        self.assertFalse((self.fixture / 'frames' / front.DATASET).exists())
        unsuitable.write_bytes(original)
        shared.install(self.fixture, self.archives, front.DATASET, release_sha256=digest)
        case = shared.cases(self.fixture, release_sha256=digest)[0]
        reference = Path(case['frame']) / 'labels.npz'
        reference.chmod(0o666)
        reference.write_bytes(b'tamper')
        with self.assertRaisesRegex(ValueError, 'Fixture file checksum mismatch'):
            shared.cases(self.fixture, release_sha256=digest)

    def test_structure_archive_is_verified_before_staging_and_on_reuse(self):
        digest = front.package(self.batches, self.release)
        structure = self.release / front.STRUCTURES
        expected = structure.read_bytes()
        structure.write_bytes(expected + b'tamper')
        with self.assertRaisesRegex(ValueError, 'Structures archive fingerprint changed'):
            shared.install(self.fixture, self.archives, front.DATASET, release_sha256=digest)
        self.assertFalse((self.fixture / 'frames' / front.DATASET).exists())
        structure.write_bytes(expected)
        shared.install(self.fixture, self.archives, front.DATASET, release_sha256=digest)
        staged = self.fixture / 'frames' / front.DATASET / front.STRUCTURES
        staged.chmod(0o666)
        staged.write_bytes(expected + b'tamper')
        with self.assertRaisesRegex(ValueError, 'Structures archive fingerprint changed'):
            shared.install(self.fixture, self.archives, front.DATASET, release_sha256=digest)

    def test_incomplete_pairs_and_changed_cameras_are_rejected(self):
        camera = dict(id='front-001-arcade-decorative-noon', clock_ticks=front.CLOCK['noon'] + 1,
                      image_size=[900, 600], eye_world=[1, 0, 0], view_rotation_projection=[1] * 16)
        self.replace_asset('decorative-noon', 'camera.json', camera)
        with self.assertRaisesRegex(ValueError, 'Native camera frame or clock differs'):
            front.package(self.batches, self.release)
        camera['clock_ticks'] = front.CLOCK['noon']
        self.replace_asset('decorative-noon', 'camera.json', camera)
        with self.assertRaisesRegex(ValueError, 'Paired camera geometry changed'):
            front.package(self.batches, self.release)
        (self.batches[0] / 'frames' / 'front-001-arcade-decorative-noon' / 'frame.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing paired captures'):
            front.package(self.batches, self.release)
        self.assertFalse(self.release.exists())

    def test_review_scope_reuses_the_frozen_full_block_fixture(self):
        from benchmark.floating_local_review import fixture_scope

        self.stage()
        fixture = self.root / 'selected.json'
        selected = shared.cases(self.fixture, lighting='morning', ids=['front-001-arcade-full-morning'])
        fixture.write_bytes(front.encode(dict(cases=selected)))
        rows, label, available, decorative_available = fixture_scope(fixture)
        self.assertEqual(rows, selected)
        self.assertEqual(label, 'Floating Front 100 · Full blocks · morning')
        self.assertEqual((available, decorative_available), (2, 2))
        selected[0]['rgb_sha256'] = '0' * 64
        fixture.write_bytes(front.encode(dict(cases=selected)))
        with self.assertRaisesRegex(ValueError, 'differs from its staged capture release'):
            fixture_scope(fixture)
        fixture.write_bytes(front.encode(dict(cases=shared.cases(self.fixture, cohort='front-decorative'))))
        with self.assertRaisesRegex(ValueError, 'Front-decorative scores are in the replay summary'):
            fixture_scope(fixture)


if __name__ == '__main__':
    unittest.main()
