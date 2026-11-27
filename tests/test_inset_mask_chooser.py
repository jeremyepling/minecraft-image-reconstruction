import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import inset_mask_chooser, inset_outline_count_guard as guard
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


def region(key,mask):
    return dict(id=key,normal_camera=[0.,0.,-1.],relative_inverse_depth=.8,
                support=dict(size=list(mask.shape[::-1]),runs=row_runs(mask)),support_area=int(mask.sum()))


class InsetMaskChooserTest(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(100,80),(160,160,160))
        self.host=np.zeros((80,100),bool)
        self.host[5:75,5:95]=True
        self.back=np.zeros_like(self.host)
        self.back[20:50,30:70]=True
        host=region('host',self.host)
        kid=region('back',self.back)
        kid['parent_id']='host'
        self.before=[host]
        self.after=[kid,region('host',self.host&~self.back)]
        self.events=[dict(host='host',output='back',photo_grid_level=3)]

    def test_supplement_is_owned_by_host_and_has_unresolved_depth(self):
        with patch.object(guard,'_span',return_value=[2,3]),patch.object(guard,'_dims',side_effect=[
                {'host':[10,8],'back':[2,3]},{'host':[10,8]}]):
            result, events=guard.apply(self.image,self.before,self.after,self.events,{'P':None})
        parts={p['id']:p for p in result}
        after=native_mask(parts['back'],self.image.size)
        expected=np.zeros_like(after)
        expected[22:48,32:68]=True
        np.testing.assert_array_equal(after,expected)
        np.testing.assert_array_equal(native_mask(parts['host'],self.image.size),self.host&~expected)
        self.assertIsNone(parts['back']['relative_inverse_depth'])
        self.assertEqual(parts['back']['outline_evidence']['depth'],'unresolved')
        self.assertEqual(events[0]['pixels'],int(expected.sum()))

    def test_mismatching_back_count_and_previously_unknown_host_count_each_veto(self):
        for first,second in [({'host':[10,8],'back':[3,3]},None),
                             ({'host':[10,None],'back':[2,3]},{'host':[10,8]})]:
            with patch.object(guard,'_span',return_value=[2,3]),patch.object(guard,'_dims',side_effect=[first,second]):
                result,events=guard.apply(self.image,self.before,self.after,self.events,{})
            self.assertEqual(result,self.before)
            self.assertFalse(events)

    def test_single_wall_needs_no_camera_or_mask_fitting(self):
        self.assertFalse(inset_mask_chooser.possible(self.image.size,self.before))
        self.assertEqual(inset_mask_chooser.supplemental(self.image,self.before,None,self.before,[]),(self.before,[]))
