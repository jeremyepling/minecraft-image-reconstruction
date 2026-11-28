from collections import Counter
import unittest

from training.floating_combined import V1, V2, check_splits, load, select_active_cases
from training.floating_cohort import frame_index


class FloatingCombinedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load()
        cls.rows = cls.manifest['frames']

    def test_canonical_active_selection_matches_frozen_policy(self):
        active = {(row['dataset'], row['id']) for row in self.rows if row['role'] == 'active'}
        self.assertEqual(active, {key for key, row in frame_index().items() if row['active']})
        self.assertEqual(self.manifest['active_splits'], dict(train=164, validation=38, test=38))
        self.assertEqual(self.manifest['sources'][V1]['aliases'], 144)
        self.assertFalse(any(row['dataset'] == V1 for row in self.rows))
        check_splits(self.rows)

    def test_unique_sources_and_paired_splits(self):
        self.assertEqual(len({row['rgb_sha256'] for row in self.rows}), len(self.rows))
        index = {(row['dataset'], row['id']): row for row in self.rows}
        for row in self.rows:
            if 'parent_id' in row:
                parent = index[row['parent_dataset'], row['parent_id']]
                self.assertEqual(row['split'], parent['split'])
                if row['role'] == 'lighting':
                    self.assertEqual(row['parent_dataset'], V2)
                    self.assertIn(parent['role'], ('active', 'regression'))
        self.assertEqual(Counter(row['role'] for row in self.rows), self.manifest['roles'])

    def test_default_selector_ignores_special_tracks(self):
        passed = list(select_active_cases(self.rows))
        self.assertEqual(len(passed), 240)
        self.assertTrue(all(row['role'] == 'active' and row['palette'] != 'orchid' for row in passed))
