import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.camera_round import digest, read
from benchmark.score_saved_planes import ROUND, candidate_manifest, score_case, select_case


class SavedPlaneScoreTests(unittest.TestCase):
    def test_two_floor_views_with_identical_pixels_use_the_original_image_path(self):
        rows = [r for r in read(ROUND / 'cases.json') if r['id'] == 'dev-wing-masonry-close-front']
        self.assertEqual(len(rows), 2)
        manifest = {'image_sha256': rows[0]['sha256']}
        with self.assertRaisesRegex(ValueError, 'identical pixels'):
            select_case(ROUND, rows, None, manifest)
        for row in rows:
            self.assertIs(select_case(ROUND, rows, ROUND / row['source'], manifest), row)

    def test_candidate_identity_requires_a_valid_manifest_or_an_explicit_image(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'example.png'
            image.write_bytes(b'image identity')
            row = {'sha256': digest(image)}
            prediction = root / 'surfaces.json'
            self.assertIsNone(candidate_manifest(prediction))
            with self.assertRaisesRegex(ValueError, 'Supply --image'):
                select_case(root, [row], None, None)
            self.assertIs(select_case(root, [row], image, None), row)
            manifest = dict(version='26.3', through='surfaces', artifact='surfaces.json', image_sha256=row['sha256'])
            (root / 'build-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertIs(select_case(root, [row], None, candidate_manifest(prediction)), row)
            image.write_bytes(b'another image')
            with self.assertRaisesRegex(ValueError, 'differs'):
                select_case(root, [row], image, manifest)

    def test_empty_candidate_is_counted_even_when_the_archived_method_found_planes(self):
        row = read(ROUND / 'cases.json')[0]
        original = score_case(ROUND, row)
        candidate = score_case(ROUND, row, [])
        self.assertGreater(original['matched'], 0)
        self.assertEqual(candidate['eligible'], original['eligible'])
        self.assertEqual((candidate['proposed'], candidate['matched']), (0, 0))
