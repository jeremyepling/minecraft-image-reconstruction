import unittest

from benchmark.floating_stage_report import capture, group, markdown


def scored_arm(selected, correct, misses, extras, *, unresolved=False):
    counts = dict(correct=correct, missed=misses, extras=extras)
    return dict(selected={k: [0, 0, 0] for k in selected},
                pack='unavailable/reconstruction.zip', pack_sha256=str(correct),
                exported=correct + extras, nonempty_unresolved=unresolved,
                score={m: dict(strict=counts, facade=counts) for m in ('geometry', 'scores')})


def comparison(before, after):
    return dict(id='arcade', dataset='visible-builds-v1', source_sha256='source', reference_sha256={},
                arms={'split-border': before, 'current': after},
                paired=dict(gain=154, loss=28, add=0, remove=0))


class FloatingStageReportTest(unittest.TestCase):
    def test_archived_scores_and_component_swap_without_live_stage_files(self):
        raw = comparison(scored_arm(['mask-2'], 28, 203, 0),
                         scored_arm(['mask-1~p1', 'horizontal-mask-0'], 154, 77, 0))
        frame = capture(raw, 'split-border', ['current'])
        self.assertEqual(frame['changes']['current']['dropped'], ['mask-2'])
        self.assertTrue(frame['changes']['current']['disjoint'])
        stats = group([frame], ('split-border', 'current'), available=144)
        self.assertEqual(stats['scores']['current']['geometry_strict']['f1_percent'], 80)
        self.assertEqual(stats['component_changes']['current']['paired']['loss'], 28)
        report = markdown({'Visible Builds': stats}, {'split-border': 'Baseline', 'current': 'Candidate 1'})
        self.assertIn('Unmeasured', report)
        self.assertIn('1/144', report)

    def test_split_surfaces_share_parent_and_unresolved_exports_get_no_credit(self):
        raw = comparison(scored_arm(['mask-1'], 28, 203, 4),
                         scored_arm(['mask-1~p24'], 154, 77, 1, unresolved=True))
        frame = capture(raw, 'split-border', ['current'])
        self.assertEqual(frame['changes']['current']['dropped'], [])
        self.assertIsNone(frame['changes']['current']['paired'])
        self.assertEqual(frame['arms']['current']['scores']['geometry_strict'],
                         dict(correct=0, missed=231, extras=155))
