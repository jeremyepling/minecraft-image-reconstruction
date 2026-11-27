import unittest

import numpy as np

from benchmark.typed_structures import aligned_sites, grade_view, geometry_score, pooled_scores, score, typed_pack


class TypedStructuresTest(unittest.TestCase):
    def test_wrong_required_id_costs_one_miss_and_one_extra(self):
        a,b,c,d = [(n,0,0) for n in range(4)]
        truth = {a:'minecraft:bricks', b:'minecraft:stone', c:'minecraft:dirt'}
        predicted = {a:'minecraft:copper_block', b:'minecraft:oak_planks', d:'minecraft:dirt'}
        result = score(predicted, truth, {b}, {a,b})
        self.assertEqual(result['strict'], dict(correct=0, missed=3, extras=3, f1=0.))
        self.assertEqual(result['facade'], dict(correct=0, missed=2, extras=2, f1=0.))
        self.assertEqual(result['material'], dict(correct=0, positions=2, accuracy=0.))
        self.assertEqual(result['assigned_positions'], 2)
        del predicted[b]
        self.assertEqual(score(predicted, truth, {b}, {a})['facade'], result['facade'])

    def test_empty_geometry_and_fallback(self):
        p = (0,0,0)
        some = score({p:'minecraft:smooth_stone'}, {p:'minecraft:smooth_stone'}, set(), set())
        none = score({}, {}, set(), set())
        self.assertEqual(some['material']['accuracy'], 1)
        self.assertEqual(some['assigned_correct'], 0)
        self.assertEqual(some['fallback'], 1)
        self.assertIsNone(none['material']['accuracy'])
        self.assertEqual(pooled_scores([none,some]), some)

    def test_reads_back_real_template_states_using_fixed_reflected_alignment(self):
        cells = [(-33,0,0),(1,0,0),(40,1,0)]
        transform = np.diag([-1,1,1,1])
        aligned = aligned_sites(cells, transform, [3,4,-2])
        self.assertEqual(aligned, [(35,4,-2),(1,4,-2),(-38,5,-2)])
        names = ['oak_planks','smooth_stone','stone_bricks']
        truth = {w:'minecraft:'+n for w,n in zip(aligned, names)}
        geometry = geometry_score(truth, truth, set())
        ref = dict(aligned=aligned, geometry=geometry, truth=[[*w,n] for w,n in truth.items()], optional=[])
        scored, pack, rows, confusions = grade_view(dict(cells=cells), ref,
               list(zip(names, ['source-atlas','fallback','source-atlas'])))
        self.assertEqual(scored['strict'], geometry['strict'])
        self.assertEqual(scored['assigned_correct'], 2)
        self.assertEqual(scored['material'], dict(correct=3, positions=3, accuracy=1))
        self.assertEqual(pack, typed_pack(dict(zip(cells, names)))[0])
        self.assertEqual(len(confusions), 3)
        self.assertEqual(rows[1], [1,0,0,'minecraft:smooth_stone','fallback'])


if __name__ == '__main__':
    unittest.main()
