import unittest

from benchmark.grid_ranker import FEATURES, aligned, apply_guard, fit, proposal, score_frames


def grid(pair, quality):
    evidence = {'fill':.87, 'aspect_known':True, 'aspect_error':.1,
                'chart_width':96, 'chart_height':80}
    for axis in ('u','v'):
        evidence.update({'agreement_'+axis:quality, 'native_'+axis:quality,
                         'rows_'+axis:32, 'transfer_'+axis:False})
    return dict(blocks=pair, counts=pair, features=evidence)


def region():
    return dict(id=1, candidates=[grid([3,4], .4), grid([5,4], .8), grid([5,4], .8)],
                baseline_index=0, baseline={'blocks':[3,4]})


class GridRankerTests(unittest.TestCase):
    def test_duplicate_dimension_does_not_hide_margin(self):
        coefficients = [0.] * len(FEATURES)
        coefficients[FEATURES.index('repeat_weak')] = 4.
        model = dict(mean=[0.]*len(FEATURES), scale=[1.]*len(FEATURES),
                     coefficients=coefficients, intercept=-2.)
        result = proposal(region(), model)
        self.assertEqual(apply_guard(result, {'threshold':.65, 'margin':.1}), 1)
        self.assertIsNone(apply_guard(result, {'threshold':.9, 'margin':.1}))

    def test_fitting_rejects_regression(self):
        with self.assertRaisesRegex(ValueError,'development'):
            fit([{'layout':'wing','cohort':'regression'}], [])

    def test_scores_keep_missed_surfaces_in_denominator(self):
        meta = dict(key='view', sha256='example', layout='wing', cohort='development')
        frame = dict(**meta, surfaces=[region()])
        target = dict(**meta, eligible=3, surfaces=[dict(id=1, expected=[5,4])])
        picks = dict(key='view', sha256='example', surfaces=[dict(id=1, index=1)])
        scores = score_frames([frame], [target], [picks])
        self.assertEqual((scores['correct'], scores['gained'], scores['missing']), (1,1,2))
        with self.assertRaisesRegex(ValueError, 'sha256'):
            list(aligned([frame], [dict(target, sha256='different')]))


if __name__ == '__main__':
    unittest.main()
