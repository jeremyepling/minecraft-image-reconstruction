import unittest

import numpy as np
from PIL import Image

from benchmark.mask_loss_trace import STAGES, loss_stage, observe_b, orientation_oracle, overlap_table
from engine.normal_regions import detect


class LossTraceTests(unittest.TestCase):
    def test_correct_uniform_direction_is_not_surface_identity(self):
        ids = np.zeros((10, 20), np.int32)
        ids[3:7, 2:6] = 1
        ids[3:7, 10:14] = 2
        wall = dict(normal_camera=[0, 0, -1])
        truth = dict(id_map=ids, image_size=[80, 40], patches={1: wall, 2: wall})
        normals = np.zeros((40, 80, 3), np.float32)
        normals[...,2] = -1.
        islands, fractions = orientation_oracle(truth, normals, {1,2})
        self.assertEqual(fractions, {1:1., 2:1.})
        self.assertIsNone(islands[1]['assigned'])
        self.assertIsNone(islands[2]['assigned'])

    def test_duplicate_fit_candidates_do_not_count_a_surface_twice(self):
        ids = np.array([[1,1,0,2,2],[1,1,0,2,2]])
        scores = overlap_table([ids==1,ids==1],ids,{1,2})
        self.assertIsNotNone(scores[1]['assigned'])
        self.assertIsNone(scores[2]['assigned'])
        stages = {stage:dict(assigned=None) for stage in STAGES}
        stages['fit_inputs']['assigned'] = 0
        self.assertEqual(loss_stage(stages,1.), 'rejected_by_fit')
        stages['emitted_support']['assigned'] = 0
        self.assertEqual(loss_stage(stages,1.), 'distorted_by_quad')

    def test_observers_leave_algorithm_output_intact(self):
        image = Image.new('RGB',(80,60),'gray')
        normals = np.zeros((30,40,3),np.float32)
        normals[...,2] = -1.
        recorded, observed = observe_b(image, normals)
        self.assertEqual(observed, detect(image,normals))
        self.assertEqual(len(recorded['initial_groups']),1)
        self.assertIs(recorded['fit_outputs'][0], observed[0])


if __name__ == '__main__':
    unittest.main()
