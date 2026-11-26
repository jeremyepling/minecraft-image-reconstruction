from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engine import vertical_atlas_faces as faces


class VerticalAtlasTests(unittest.TestCase):
    def test_only_coplanar_neighbors_nominate_a_face(self):
        prior={(0,0,0),(0,2,0)}
        proposed=faces.candidate_faces(prior)
        self.assertEqual(proposed[((0,1,0),2)],2)
        self.assertNotIn(((0,0,1),2),proposed)
        self.assertNotIn(((0,0,0),2),proposed)

    def test_missing_atlas_or_model_preserves_prior(self):
        prior={(0,0,0)}
        self.assertEqual(faces.recover(None,[{'support':True}],{'P':None},prior,None),prior)
        with TemporaryDirectory() as directory:
            model=Path(directory)/'weights.json'
            with patch.object(faces,'WEIGHTS',model):
                for text in (None,'{}','{"weights":[1]}'):
                    if text is not None:
                        model.write_text(text)
                    with patch.object(faces,'observations',side_effect=AssertionError):
                        self.assertEqual(faces.recover(None,[],{},prior,None),prior)
