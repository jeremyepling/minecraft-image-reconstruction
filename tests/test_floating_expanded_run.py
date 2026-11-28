"""The default replay keeps the archived orientation and miss accounting."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmark.expanded_scores import FrameReference
from benchmark.floating_expanded_run import score_export
from engine.export import JAVA_26_3_FORMAT, write_pack


class ExpandedReplayScoreTests(unittest.TestCase):
    def setUp(self):
        self.truth = {(7, 1, 3): 'minecraft:stone', (7, 2, 3): 'minecraft:bricks'}
        self.reference = FrameReference(self.truth, {(7, 2, 3)}, {}, {'full_cube': set(self.truth)}, {})
        self.entry = {'id': 'view', 'reference_sha256': {},
                      'variants': {'original': {'source_sha256': 'image-hash'}}}
        self.anchor = {'normal_camera': [1, 0, 0], 'local_normal_axis': 1}

    def test_unresolved_export_charges_every_block_and_reference_miss(self):
        with TemporaryDirectory() as directory:
            pack = Path(directory) / 'reconstruction.zip'
            write_pack(pack, {(0, 0, 0): 'stone'}, JAVA_26_3_FORMAT)
            score, aligned = score_export(self.entry, pack, dict(size=1, orientation=None, assigned=[[0, 0, 0]]),
                                          reference=self.reference)
        self.assertEqual(score['registration_status'], 'unresolved')
        self.assertIsNone(score['translation'])
        self.assertEqual(score['geometry']['strict'], dict(correct=0, missed=2, extras=1, f1=0))
        self.assertEqual(score['scores']['facade'], dict(correct=0, missed=1, extras=1, f1=0))
        self.assertEqual(aligned, {})

    def test_resolved_export_uses_one_geometry_translation_for_exact_ids(self):
        with TemporaryDirectory() as directory:
            pack = Path(directory) / 'reconstruction.zip'
            write_pack(pack, {(0, 0, 0): 'stone', (0, 1, 0): 'dirt'}, JAVA_26_3_FORMAT)
            with patch('benchmark.floating_expanded_run.yaw_from_anchor_normal', return_value={'yaw_quarters': 0}):
                score, aligned = score_export(self.entry, pack, dict(size=2, orientation=self.anchor,
                                                                    assigned=[[0, 0, 0], [0, 1, 0]]),
                                              reference=self.reference)
        self.assertEqual(score['registration_status'], 'resolved')
        self.assertEqual(score['translation'], [7, 1, 3])
        self.assertEqual(score['geometry']['strict']['correct'], 2)
        self.assertEqual(score['scores']['strict']['correct'], 1)
        self.assertEqual(aligned, {(7, 1, 3): 'minecraft:stone', (7, 2, 3): 'minecraft:dirt'})
