import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmark import current_placement_viewer as viewer
from benchmark.texture_camera_eval import digest, save


class PlacementPreviewTests(unittest.TestCase):
    def test_candidate_continues_identical_source_and_export(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root/'base', root/'candidate'
            base.mkdir()
            candidate.mkdir()
            row = dict(id='view', split='test', source_hash='image', previous_pack_sha256='before',
                       pack_sha256='placed', result={'strict': 7})
            save(base/'cases.json', [row])
            save(base/'summary.json', dict(cohorts={}, cases_sha256=digest(base/'cases.json')))
            next_row = dict(row, previous_pack_sha256='placed', pack_sha256='next',
                            baseline=row['result'], result={'strict': 9})

            def write_release(change):
                save(candidate/'cases.json', [change])
                save(candidate/'summary.json', dict(previous_cohorts={}, changed=[change],
                                                   cases_sha256=digest(candidate/'cases.json')))

            with patch.object(viewer, 'DEFAULT_RELEASES', [base]):
                write_release(next_row)
                self.assertEqual(viewer.preview_chain([candidate]), [base, candidate])
                for replacement in (dict(source_hash='other'), dict(previous_pack_sha256='older')):
                    with self.subTest(replacement=replacement):
                        write_release(dict(next_row, **replacement))
                        with self.assertRaisesRegex(ValueError, 'view'):
                            viewer.preview_chain([candidate])

    def test_private_render_stays_isolated_on_failure(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)/'result'

            def fail(*args, page, assets, **kwargs):
                assets.mkdir(parents=True)
                (assets/'partial').write_bytes(b'partial')
                raise RuntimeError('unusable capture')

            with patch.object(sys, 'argv', ['viewer', '--out', str(out)]), patch.object(viewer, 'render', fail):
                with self.assertRaisesRegex(RuntimeError, 'unusable capture'):
                    viewer.main()
            self.assertFalse(out.exists())

    def test_preview_labels_are_text_and_historical_links_are_inactive(self):
        totals = dict(views=1, points=0, share=0)
        path = dict(ids=['frame'])
        gap = dict(selected=totals, layouts=totals, total=0, baseline=100., displayedCeiling=100.,
                   paths=dict(layouts=[path], images=[path]),
                   themes=dict(uncovered=0,count=0,origin=0,position=0,direction=0,ties=0))
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            viewer.write_viewer(folder, [], 'assets', '<test>&__DATA__', gap, preview=True, candidate=True)
            html = (folder/'viewer.html').read_text(encoding='utf-8')
        self.assertIn('Candidate placement <code>&lt;test&gt;&amp;__DATA__</code>', html)
        self.assertIn('href=\"assets/stage-map.jpg\"', html)
        self.assertNotIn('href=\"../pavement-local-01/openings.jpg\"', html)
        self.assertIn('already reaches 100.00%', html)
