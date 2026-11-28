"""The component ranker uses projected support and breaks ties with the incumbent."""
import unittest

from engine.floating_group_coverage import select_group


class FloatingGroupCoverageTests(unittest.TestCase):
    def test_prefers_larger_supported_footprint_to_more_cells(self):
        groups = [
            {'origins': {'tiny': [0, 0, 0]}, 'blocks': {x: 'stone' for x in range(16)}},
            {'origins': {'large': [0, 0, 0]}, 'blocks': {0: 'stone'}},
        ]
        charts = [
            {'id': 'tiny', 'H': [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
             'cells': [{'u': u, 'v': v} for u in range(4) for v in range(4)]},
            {'id': 'large', 'H': [[16, 0, 20], [0, 16, 20], [0, 0, 1]],
             'cells': [{'u': 0, 'v': 0}]},
        ]
        planes = [dict(id=chart['id'], support=dict(size=[40, 40], runs=[[y, 0, 40] for y in range(40)]))
                  for chart in charts]
        self.assertIs(select_group(groups, charts, planes, (40, 40)), groups[1])
        groups[1]['blocks'] = {x: 'stone' for x in range(5)}
        self.assertIs(select_group(groups, charts, planes, (40, 40)), groups[0])
        groups[1]['blocks'] = {0: 'stone'}
        charts[1]['export_uv'] = []
        self.assertIs(select_group(groups, charts, planes, (40, 40)), groups[0])
        charts[0]['export_uv'] = []
        self.assertIs(select_group(groups, charts, planes, (40, 40)), groups[0])
