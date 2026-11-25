import unittest
from dataclasses import replace

from benchmark.exterior_set import check_capture_conditions, load_suite, pair_cases


class ExteriorSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release, cls.cases = load_suite()

    def test_all_pairs_are_available_at_clear_noon(self):
        self.assertEqual(len(self.cases), 20)
        self.assertTrue(all(len(pair) == 2 for pair in self.cases.values()))
        self.assertEqual(self.release['capture_conditions']['clock_ticks'], 6000)
        self.assertIs(self.release['capture_conditions']['raining'], False)
        self.assertIs(self.release['capture_conditions']['thundering'], False)

    def test_reject_dusk_rain_thunder_and_missing_weather(self):
        expected = self.release['capture_conditions']
        for key, value in [('clock_ticks', 12000), ('raining', True), ('thundering', True)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                check_capture_conditions(dict(expected, **{key: value}), expected, 'altered')
        with self.assertRaisesRegex(ValueError, 'raining'):
            check_capture_conditions({k:v for k,v in expected.items() if k!='raining'}, expected, 'altered')

    def test_unpaired_or_mislabeled_floor_is_rejected(self):
        a, b = next(iter(self.cases.values())).values()
        with self.assertRaisesRegex(ValueError, 'Missing floor'):
            pair_cases({a.floor:[a], b.floor:[]})
        altered = replace(b, frame=dict(b.frame, angle='wrong-pose'))
        with self.assertRaisesRegex(ValueError, 'Mismatched identity'):
            pair_cases({a.floor:[a], b.floor:[altered]})
