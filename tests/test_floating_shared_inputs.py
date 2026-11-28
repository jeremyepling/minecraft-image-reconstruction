import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from training import floating_shared_inputs as shared


class FloatingSharedInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(TemporaryDirectory())
        self.root = Path(self.temp) / 'shared'
        self.archives = Path(self.temp) / 'archives'
        self.payloads = {name: f'original {name}'.encode() for name in shared.CAPTURE_FILES}
        self.hashes = {name: shared.hash_bytes(payload).hexdigest() for name, payload in self.payloads.items()}
        self.visible = self.source('visible-builds-v2', 'train-visible')
        self.decorative = self.source('decorative-builds-v1', 'train-decorative')
        self.enterContext(patch.object(shared, 'active_rows', return_value=[self.visible, self.decorative]))

    def source(self, dataset, name):
        archive = f'{dataset}/train.zip'
        member = f'frames/{name}'
        path = self.archives / archive
        path.parent.mkdir(parents=True)
        with ZipFile(path, 'w') as shard:
            for filename, payload in self.payloads.items():
                shard.writestr(f'{member}/{filename}', payload)
        return dict(id=name, dataset=dataset, archive=archive, member=member, files=self.hashes,
                    framing={'accepted': True}, split='train', layout=name, family=name, palette='cool',
                    rgb_sha256=self.hashes['rgb.png'])

    def install(self, row):
        record = dict(row, directory=row['member'], counts={'visible_blocks': 5}, ground_y=64)
        with patch.object(shared, 'read_release', return_value=[(record, row['archive'])]):
            return shared.install(self.root, self.archives, row['dataset'])

    def test_each_cohort_uses_its_own_pinned_frames(self):
        self.assertEqual(self.install(self.visible), 1)
        with self.assertRaisesRegex(FileNotFoundError, 'decorative-builds-v1'):
            shared.cases(self.root, cohort='decorative')
        self.assertEqual(self.install(self.decorative), 1)
        for cohort, source in [('visible', self.visible), ('decorative', self.decorative)]:
            with self.subTest(cohort=cohort):
                case, = shared.cases(self.root, cohort=cohort)
                self.assertEqual(case['id'], source['id'])
                self.assertEqual(case['dataset_manifest_sha256'], shared.PINNED[source['dataset']])
                frame = Path(case['frame'])
                self.assertEqual(frame / 'rgb.png', Path(case['variants']['original']['image']))
                self.assertEqual(json.loads((frame / 'frame.json').read_bytes())['counts'], {'visible_blocks': 5})
        with self.assertRaisesRegex(ValueError, 'Unknown visible views'):
            shared.cases(self.root, cohort='visible', ids=[self.decorative['id']])

    def test_checksum_failure_does_not_publish_dataset(self):
        with ZipFile(self.archives / self.visible['archive'], 'w') as shard:
            for name, payload in self.payloads.items():
                shard.writestr(f'{self.visible["member"]}/{name}', b'changed' if name == 'rgb.png' else payload)
        with self.assertRaisesRegex(ValueError, 'Capture file checksum mismatch'):
            self.install(self.visible)
        self.assertFalse((self.root / 'frames' / self.visible['dataset']).exists())


if __name__ == '__main__':
    unittest.main()
