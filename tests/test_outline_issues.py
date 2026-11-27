import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.outline_issues import outline_issues
from engine.plane_consensus import row_runs


class OutlineIssuesTest(unittest.TestCase):
    def test_straight_diagonal_and_repeated_large_turns(self):
        def flags(points):
            image=Image.new('1',(200,90))
            ImageDraw.Draw(image).polygon(points,fill=1)
            part=dict(id='patch',support=dict(size=list(image.size),runs=row_runs(np.asarray(image))))
            return outline_issues([part],image.size)
        self.assertEqual([],flags([(8,5),(172,27),(164,74),(8,60)]))
        teeth=[(8+i*7,15+(i%2)*10) for i in range(23)]
        issues=flags(teeth+[(162,75),(8,75)])
        self.assertEqual('patch',issues[0]['surface'])
        self.assertGreaterEqual(len(issues[0]['native_turn_xy']),8)
