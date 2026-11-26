import unittest

import numpy as np

from benchmark.phase_material_round import grade, link


class MaterialFaceScoreTest(unittest.TestCase):
    def test_correct_id_requires_association_and_aligned_corners(self):
        valid = {(1,(0,0)):dict(block_id='stone_bricks',xyz=[0,0,0],direction='north'),
                 (1,(1,0)):dict(block_id='stone_bricks',xyz=[1,0,0],direction='north')}
        def prediction(left,guess):
            return dict(proposal='wall',cell=[0,0],quad=[[left,0],[left+1,0],[left+1,1],[left,1]],
                        block_id='minecraft:'+guess,block_index=0,material_source_face=0)
        rows = [prediction(0,'bricks'),prediction(1.3,'stone_bricks'),prediction(8,'stone_bricks')]
        links = link(rows,{'wall':1},valid,{'charts':{1:np.eye(3)}})
        scores,records = grade(rows,links,valid)
        self.assertEqual(records[0]['reference'],[1,0,0])
        self.assertEqual((scores['wrong_id'],scores['misaligned'],scores['unpaired']),(1,1,1))
        self.assertEqual((scores['correct'],scores['unverified_named']),(0,2))
