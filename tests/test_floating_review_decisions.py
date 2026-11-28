"""The archived, source-matched V1 comparison is the review contract fixture."""
import copy
import json
from pathlib import Path
import unittest

from benchmark.floating_perfect_report import EARLY, from_archive
from benchmark.floating_review_decisions import analyze
from benchmark.floating_stage_report import read


REVIEWS = Path(__file__).resolve().parents[1] / 'benchmark' / 'reviews'


class FloatingDecisionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = read(REVIEWS / 'floating-stage-report-01/report.json')
        cls.receipts = read(REVIEWS / 'floating-auto-review-queue-01/receipts.json')
        controls = read(REVIEWS / 'floating-perfect-stage-01/controls.json.gz')
        early = {stage: read(REVIEWS / 'floating-perfect-stage-inputs-01' / f'{stage}-conditions.json.gz')
                 for stage in EARLY}
        cls.report['perfect_stage'] = from_archive(controls, cls.report['comparison_sha256'],
                                                    cls.report['frames'], cls.report['names'], early)
        directory = REVIEWS / 'mobile-review-assets/floating-v1-guarded-refit'
        cls.evidence = {case['id']: read(directory / case['id'] / 'surfaces.json') for case in cls.receipts['cases']}

    def test_same_cohort_placement_headroom_and_independent_work(self):
        result = analyze(self.report, self.receipts)
        first = result['actions'][0]
        self.assertEqual((first['id'], first['diagnosticCategory']), ('replay-grid', 'exact-input'))
        self.assertIn('+14,107', first['evidence'])
        self.assertIn('-6,547', first['evidence'])
        self.assertEqual([g['tested'] for g in result['diagnosticGroups']], [144, 12])
        self.assertNotEqual(*[g['frameSetSha256'] for g in result['diagnosticGroups']])
        rows = {r['stage']: r for r in result['diagnosticGroups'][0]['rows']}
        self.assertEqual(rows['Local heading']['category'], 'bounded-chooser')
        self.assertEqual(rows['Pack writing']['category'], 'writer-check')
        self.assertAlmostEqual(rows['Block Counter']['delta']['geometry_strict'], 42.67, places=2)
        self.assertEqual(result['diagnosticsMissing'], ['Relative joins', 'Component selection'])
        self.assertFalse(any(card['disposition'] == 'question' for card in result['actions']))
        cards = {card['id']: card for card in result['actions']}
        self.assertEqual(cards['audit-full-face-grid']['caseIds'], ['train-sawtooth1-cool-low-left'])
        self.assertEqual(len(cards['audit-unresolved']['caseIds']), 7)
        self.assertEqual(len(cards['audit-component-swap']['caseIds']), 2)
        self.assertNotIn('replay-writer', cards)

    def test_existing_export_oracle_and_held_out_pilot_gate(self):
        result = analyze(self.report, self.receipts, surface_evidence=self.evidence)
        pilot = result['chooserPilot']
        self.assertEqual(pilot['exportChooser']['tested'], 144)
        self.assertEqual(pilot['exportChooser']['switches'], 0)
        self.assertEqual(pilot['exportChooser']['before'], pilot['exportChooser']['after'])
        self.assertEqual(pilot['headingBySplit']['train'], dict(views=72, unresolved=0, correctGain=0))
        self.assertEqual(pilot['headingBySplit']['test'], dict(views=12, unresolved=0, correctGain=0))
        self.assertEqual(sum(row['correctGain'] for row in pilot['headingBySplit'].values()), 617)
        self.assertFalse(pilot['gridChooserMeasured'])
        measured = pilot['reviewedGrids']
        self.assertEqual((measured['parents'], measured['multipleGridLayers'], measured['splits']), (59, 25, {'train': 4}))
        self.assertEqual(measured['families'], ['arcade', 'sawtooth', 'terrace'])

    def test_mismatched_diagnostic_or_queue_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['perfect_stage']['rows'][0]['cases'][0]['source_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'frame receipt'):
            analyze(report, self.receipts)
        queue = copy.deepcopy(self.receipts)
        queue['cases'][0]['paired']['loss'] += 1
        with self.assertRaisesRegex(ValueError, 'Paired review counts'):
            analyze(self.report, queue)

    def test_equal_counts_do_not_pool_different_views(self):
        report = copy.deepcopy(self.report)
        full = report['perfect_stage']['rows'][0]
        from benchmark.floating_perfect_report import aggregate
        rows = []
        for case in full['cases'][:2]:
            rows.append(dict(full, cohort='Subset', tested=1, cases=[case],
                             before=aggregate([case['before']]), after=aggregate([case['after']])))
        report['perfect_stage']['rows'] = rows
        result = analyze(report, self.receipts)
        self.assertEqual(len({g['frameSetSha256'] for g in result['diagnosticGroups']}), 2)

    def test_missing_diagnostics_trigger_measurement_and_verified_source_counts(self):
        report = dict(self.report, perfect_stage=None)
        result = analyze(report, self.receipts)
        self.assertFalse(result['diagnosticGroups'])
        card = next(c for c in result['actions'] if c['id'] == 'measure-grid')
        self.assertIn('412 / 2,052', card['evidence'])
        self.assertEqual(card['basis'], 'unmeasured')

    def test_single_arm_never_borrows_older_v1_choosers(self):
        report = copy.deepcopy(self.report)
        report.update(names=dict(normal='Combined V1 normal runner'), perfect_stage=None, cohort_label='Combined V1')
        frame = report['frames'][0]
        frame['arms'] = dict(normal=frame['arms']['current'])
        frame['changes'] = {}
        report['frames'] = [frame]
        report['cohorts']['Visible Builds'] = dict(views=1, stages=dict(normal={}))
        result = analyze(report)
        pilot = result['chooserPilot']
        self.assertEqual(pilot['exportChooser']['status'], 'unmeasured')
        self.assertIsNone(pilot['exportChooser']['after'])
        self.assertFalse(pilot['headingBySplit'])
        self.assertFalse(result['diagnosticGroups'])
        self.assertIn('Block Counter', result['diagnosticsMissing'])
        report['perfect_stage'] = self.report['perfect_stage']
        with self.assertRaisesRegex(ValueError, 'matching source-only arm'):
            analyze(report)

    def test_surface_overlap_needs_matched_packs_and_does_not_claim_material_error(self):
        case = self.receipts['cases'][0]
        marker = case['id']
        event = dict(chart='roof', uv=[1, 2], xyz=[0, 0, 2], block='same', winnerChart='wall', winnerBlock='same')
        package = dict(schema=2, id=marker, sourceSha256=case['source_sha256'], provenance=dict(arms={
            'Baseline': dict(packSha256=case['baseline']['pack_sha256']),
            'Candidate': dict(packSha256=case['candidate']['pack_sha256'])}),
            surfaces=[dict(id='roof', layers=[], exports={'Candidate': dict(overlaps=[event])})])
        result = analyze(self.report, self.receipts, surface_evidence={marker: package})
        action = next(c for c in result['actions'] if c['id'] == 'audit-chart-ownership')
        self.assertIn('roof lost [0, 0, 2] to wall', action['evidence'])
        self.assertEqual(action['disposition'], 'machine')
        package['provenance']['arms']['Candidate']['packSha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Candidate export'):
            analyze(self.report, self.receipts, surface_evidence={marker: package})

    def test_consequential_questions_capped_deduplicated_and_answered_privately(self):
        case = self.receipts['cases'][0]
        question = dict(id='ask-corner', caseId=case['id'], sourceSha256=case['source_sha256'],
                        theme='corner', stage='Relative joins', type='visible-ambiguity',
                        prompt='Which edge continues behind the rim?', cue='Use the highlighted intersection.',
                        checked='Both candidate joins reach the same visible edge in this view.',
                        firstPass='The short edge appears continuous.',
                        choices=[dict(answer='Short edge', experiment='Test preserving the short offset.'),
                                 dict(answer='Long edge', experiment='Test the long-edge join.')])
        gates = [dict(question, id=f'ask-{i}', theme=theme) for i, theme in enumerate(('corner', 'corner', 'depth', 'pitch', 'material'))]
        result = analyze(self.report, self.receipts, ledger=dict(questions=gates))
        self.assertEqual([c['id'] for c in result['actions'] if c['disposition'] == 'question'], ['ask-0', 'ask-2', 'ask-3'])
        entry = dict(actionId='ask-0', caseId=case['id'], sourceSha256=case['source_sha256'], questionType='visible-ambiguity',
                     status='tested', answer='private answer', text='private note', outcome='private outcome',
                     changedExperiment=True, measuredEffect=True, measuredViews=12, measuredComparison='a'*64,
                     measuredDelta=dict(correct=2, missed=-2, extras=-1, geometry_strict=1.5, material_strict=1.1))
        result = analyze(self.report, self.receipts, ledger=dict(questions=gates, entries=[entry, entry]))
        self.assertNotIn('ask-0', [c['id'] for c in result['actions']])
        self.assertEqual(result['questionMetrics']['byType']['visible-ambiguity'], dict(
            answered=1, tested=1, changedExperiment=1, measuredEffects=1, gainedCorrect=1, reducedExtras=1, gainedMaterial=1))
        self.assertEqual(result['outcomes'][0]['measured']['delta']['correct'], 2)
        self.assertNotIn('private', json.dumps(result))
        entry['sourceSha256'] = 'old'
        result = analyze(self.report, self.receipts, ledger=dict(questions=gates, entries=[entry]))
        self.assertIn('ask-0', [c['id'] for c in result['actions']])


if __name__ == '__main__':
    unittest.main()
