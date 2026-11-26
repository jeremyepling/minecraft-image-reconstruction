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

    def test_final_verifier_rechecks_only_after_a_change(self):
        first = {(0, 0, 0)}
        second = first | {(0, 1, 0)}
        with patch.object(faces, 'recover', side_effect=(second, second)) as checked:
            self.assertEqual(faces.confirm_after_roof(None, [{}], {'P': 1}, first, object()), second)
        self.assertEqual([c.args[3] for c in checked.call_args_list], [first, second])
        self.assertTrue(all(c.kwargs['mode'] == .95 for c in checked.call_args_list))

    def test_final_verifier_preserves_missing_inputs_and_caps_iterations(self):
        prior = {(0, 0, 0)}
        with patch.object(faces, 'recover', side_effect=AssertionError):
            self.assertEqual(faces.confirm_after_roof(None, [{}], {'P': 1}, prior, None), prior)
            self.assertEqual(faces.confirm_after_roof(None, [{}], {'P': None}, prior, object()), prior)
        def grow(image, parts, camera, cells, atlas, mode):
            return cells | {(0, len(cells), 0)}
        with patch.object(faces, 'recover', side_effect=grow) as checked:
            self.assertEqual(len(faces.confirm_after_roof(None, [{}], {'P': 1}, prior, object())), 4)
            self.assertEqual(checked.call_count, 3)
