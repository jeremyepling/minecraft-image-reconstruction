from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote
import unittest

from benchmark.round_report import FIXTURE, paired_table, perfect_repairs, pooled, report, validate


class RoundReportTest(unittest.TestCase):
    def setUp(self):
        score = dict(correct=10, missed=2, extras=1)
        result = {metric: score.copy() for metric in ('strict', 'facade', 'registered')}
        self.reference = {'view': dict(id='view', source_hash='original', split='test', result=result,
                                       previous_pack_sha256='old', pack_sha256='current')}
        self.trial = deepcopy(self.reference)

    def test_same_id_requires_the_same_image_and_targets(self):
        self.trial['view']['source_hash'] = 'another'
        with self.assertRaisesRegex(ValueError, 'source image'):
            validate(self.trial, self.reference, 'result')
        self.trial['view']['source_hash'] = 'original'
        self.trial['view']['result']['facade']['missed'] -= 1
        with self.assertRaisesRegex(ValueError, 'target count'):
            validate(self.trial, self.reference, 'result')

    def test_net_ties_can_hide_coordinate_churn(self):
        row = self.trial['view']
        row.update(previous_pack_sha256='current', pack_sha256='new',
                   block_deltas=dict(gained=2, lost=2, extra_added=1, extra_removed=1))
        description = paired_table(['view'], self.reference, 'result', self.trial)
        self.assertIn('Correct gained/lost 2/2', description)
        row['previous_pack_sha256'] = 'different'
        self.assertIn('0/1 views', paired_table(['view'], self.reference, 'result', self.trial))

    def test_perfect_repair_uses_the_pooled_target(self):
        good = dict(correct=99, missed=0, extras=0)
        poor = dict(correct=1, missed=0, extras=11)
        views = [good, poor]
        self.assertEqual(perfect_repairs(views, pooled(views)), 1)
        self.assertEqual(perfect_repairs([good], good), 0)

    def test_denominator_and_relative_evidence_link_in_release_report(self):
        output_dir = Path.cwd() / 'runs' / 'a nested report'
        result = report(FIXTURE, link_from=output_dir)
        self.assertIn('1.5806 percentage points remain', result)
        target = result.split('[Release findings and evidence](', 1)[1].split(')', 1)[0]
        self.assertEqual((output_dir / unquote(target)).resolve(), FIXTURE / 'README.md')
