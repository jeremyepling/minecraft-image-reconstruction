import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import facade_inset_phase as phase
from engine.facade_inset_phase import back_surface, projected_back_area, rectangles, refine, supplemental, texel_bins
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


class FacadePhaseTest(unittest.TestCase):
    def test_candidate_requires_closed_mouth_on_four_sides(self):
        weak={(x,y) for x in range(4) for y in range(3)}
        ring={(x,y) for x in range(-1,5) for y in range(-1,4)}-weak
        self.assertEqual(len(list(rectangles(ring,weak))),1)
        self.assertFalse(list(rectangles({p for p in ring if p[1]!=3},weak)))
        hole=(1,1)
        self.assertFalse(list(rectangles(ring|{hole},weak-{hole})))

    def test_supported_deeper_plane_and_abstention_on_flush_art_or_fragment(self):
        eye=np.array([-2.,6.,-8.])
        linear=np.array([[200.,0,140],[0,-200.,150],[0,0,1.]])
        p=np.column_stack((linear,-linear@eye))
        y,x=np.mgrid[70:300,150:345]
        ray=np.stack((x.ravel()+.5,y.ravel()+.5,np.ones(x.size)),axis=1)@np.linalg.inv(linear).T
        ratio=ray[:,:2]/ray[:,2,None]
        uv=eye[:2]+8*ratio
        lo,hi=np.array([1,2]),np.array([5,5])
        inside=((uv>=lo)&(uv<hi)).all(axis=1)
        ratio,uv=ratio[inside],uv[inside]
        rows={}
        tile=np.floor(uv).astype(int)
        for a,b in set(map(tuple,tile)):
            rows[a,b]=np.flatnonzero((tile==[a,b]).all(axis=1))
        template=np.random.default_rng(7).normal(100,20,256)
        data=dict(uv=uv,ratio=ratio,plane=0,eye=eye,axis=2,free=[0,1],p=p,fit=dict(corr=1.,margin=.90))
        back=uv+ratio
        gray=template[texel_bins(back)]
        actual=back_surface(data,set(rows),lo,hi,rows,gray,template)
        self.assertIsNotNone(actual)
        support,meta=actual
        self.assertEqual(meta['depth'],1)
        self.assertEqual(support.sum(),projected_back_area(data,lo,hi,1))
        # A different brightness at exactly the front texture phase remains a single wall.
        self.assertIsNone(back_surface(data,set(rows),lo,hi,rows,.6*template[texel_bins(uv)],template))
        # A small remaining piece of a backing largely owned by another mask cannot create a second mask.
        small={k:v[::3] for k,v in rows.items()}
        self.assertIsNone(back_surface(data,set(small),lo,hi,small,gray,template))

    def test_invalid_camera_returns_unmodified_surface_objects(self):
        photo=Image.new('RGB',(100,100))
        parts=[dict(id='wall')]
        for camera in (None,{'P':None},{'P':np.ones((3,4))},{'P':'broken'}):
            output,events=refine(photo,parts,camera)
            self.assertIs(parts,output)
            self.assertEqual(events,[])

    def test_supplement_checks_only_witnessed_host_and_preserves_earlier_children(self):
        image=Image.new('RGB',(24,24))
        foreground=np.ones((24,24),bool)
        host=dict(id='wall',parent_id='before-split',normal_camera=[0,0,1],support=dict(size=[24,24],runs=row_runs(foreground)))
        previous=dict(id='earlier-back',parent_id='wall',metadata={'signed_depth':1})
        unrelated=dict(id='another-wall',normal_camera=[0,0,1],support=host['support'])
        parts=[previous,host,unrelated]
        event=dict(host='wall',side='jamb',sill='sill',output='earlier-back',level=0)
        projection=np.column_stack((np.eye(3),np.zeros(3)))
        cam={'P':projection}
        y,x=np.nonzero(foreground)
        cut=(x>5)&(x<18)&(y>5)&(y<18)
        with patch.object(phase,'host_grid',return_value=dict(y=y,x=x,axis=2,plane=0)) as grid, \
                patch.object(phase,'tiles',return_value=({},None,None)), \
                patch.object(phase,'front_anomalies',return_value=(set(),set())), \
                patch.object(phase,'rectangles',return_value=[(None,None,None)]), \
                patch.object(phase,'back_surface',return_value=(cut,dict(depth=1))):
            self.assertIs(supplemental(image,parts,cam,[])[0],parts)
            self.assertIs(supplemental(image,parts,cam,[dict(host='wall',side='jamb')])[0],parts)
            result,events=supplemental(image,parts,cam,[event])
            self.assertEqual(grid.call_count,1)
            self.assertIs(grid.call_args.args[1],host)
            self.assertIs(result[0],previous)
            self.assertIs(result[-1],unrelated)
            by_id={part['id']:part for part in result}
            back=by_id[events[0]['output']]
            self.assertEqual(back['parent_id'],host['id'])
            self.assertEqual(by_id['wall']['parent_id'],'before-split')
            mask=native_mask(back,image.size)
            left=native_mask(by_id['wall'],image.size)
            np.testing.assert_array_equal(left|mask,foreground)
            self.assertFalse((mask&left).any())
            self.assertIs(supplemental(image,result,cam,[event])[0],result)


if __name__=='__main__':
    unittest.main()
