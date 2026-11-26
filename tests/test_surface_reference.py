"""Checks for the published, evaluation-only reference geometry."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import numpy as np

from benchmark.camera_round import ROOT, read
from benchmark.exterior_set import Capture
from benchmark.surface_metrics import truth
from benchmark.surface_reference import ARCHIVE, build, recorded, review_case, verify


class SavedSurfaceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = ROOT / 'benchmark/reviews/normals-01'
        cls.cases = [review_case(run, row) for row in read(run / 'cases.json')]
        cls.case = next(c for c in cls.cases if c.id == 'holdout-galleries-masonry-low-left')

    def test_all_records_are_present_and_bound_to_the_sources(self):
        self.assertEqual(len(self.cases), 56)
        verify(self.cases)

    def test_archived_geometry_agrees_with_original_projection(self):
        case = self.case
        pose, cache = recorded(case)
        fresh = truth(pose, read(case.path('labels'))['faces'])
        self.assertEqual(cache['image_size'], fresh['image_size'])
        np.testing.assert_array_equal(cache['id_map'], fresh['id_map'])
        self.assertEqual(cache['patches'].keys(), fresh['patches'].keys())
        self.assertEqual([r['patch_id'] for r in cache['rectangles']], [r['patch_id'] for r in fresh['rectangles']])
        for i, patch in fresh['patches'].items():
            actual = cache['patches'][i]
            for key in ('id', 'axis', 'normal', 'offset', 'faces', 'support_px', 'eligible'):
                self.assertEqual(actual[key], patch[key])
            np.testing.assert_allclose(actual['normal_camera'], patch['normal_camera'], atol=1e-10)
            np.testing.assert_allclose(cache['charts'][i], fresh['charts'][i], atol=1e-10)
            np.testing.assert_allclose(cache['vertices'][i], fresh['vertices'][i], atol=1e-10)
            rect = patch['rectangle']
            if rect is None:
                self.assertIsNone(actual['rectangle'])
            else:
                self.assertEqual(actual['rectangle']['block_bounds'], rect['block_bounds'])
                for key in ('counts', 'quad'):
                    np.testing.assert_allclose(actual['rectangle'][key], rect[key], atol=1e-10)

    def test_stale_labels_and_corrupt_geometry_are_rejected(self):
        case = self.case
        with TemporaryDirectory() as directory:
            temp = Path(directory)
            copy = Capture(case.floor, temp, case.frame)
            for key in ('image', 'pose', 'states', 'labels'):
                copy.path(key).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(case.path(key), copy.path(key))
            with copy.path('labels').open('ab') as src:
                src.write(b' ')
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                recorded(copy)

            member = f'{case.floor}/{case.id}.json'
            corrupt = temp / 'broken.zip'
            with ZipFile(ARCHIVE) as original, ZipFile(corrupt, 'w') as edited:
                edited.writestr('index.json', original.read('index.json'))
                edited.writestr(member, b'{}')
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                recorded(case, corrupt)

    def test_archive_rebuild_and_version_checks(self):
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'geometry.zip'
            pose, direct = recorded(self.case, archive)
            build([self.case], archive)
            _, loaded = recorded(self.case, archive)
            np.testing.assert_array_equal(direct['id_map'], loaded['id_map'])
            with self.assertRaisesRegex(ValueError, 'membership differ'):
                verify(self.cases, archive)
            with ZipFile(archive) as original:
                index = json.loads(original.read('index.json'))
            index['generator_sha256'] = 'obsolete'
            with ZipFile(archive, 'w') as edited:
                edited.writestr('index.json', json.dumps(index))
            with self.assertRaisesRegex(ValueError, 'incompatible geometry version'):
                recorded(self.case, archive)
