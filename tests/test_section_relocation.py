from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import section_decision as model


class SectionRelocationTests(TestCase):
    def test_guard_only_accepts_deeper_already_covered_candidate(self):
        image=Image.new('RGB',(96,96))
        matrix=np.array([[95.,0,48],[0,95,48],[0,0,1]])
        eye=np.array([.5,.5,-3])
        P=np.column_stack((matrix,-matrix@eye))
        original={(0,0,0)}
        deeper,closer,uncovered=(0,0,2),(0,0,-1),(2,0,2)
        self.assertEqual(model.deeper_cells(image,original,{closer,deeper,uncovered},P,eye),{deeper})

    def test_model_fingerprint_and_fallback(self):
        raw=model.WEIGHTS.read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(model.read_model(raw),model.read_model(raw.replace(b'\n',b'\r\n')))
        self.assertIsNone(model.read_model(b'{}'))
        existing={(0,0,0)}
        with TemporaryDirectory() as temp, patch.object(model,'sections') as measure:
            file=Path(temp)/'wrong.json'
            with patch.object(model,'WEIGHTS',file):
                self.assertEqual(model.recover(None,[],{},existing,mode='deeper'),existing)
                file.write_bytes(b'{}')
                self.assertEqual(model.recover(None,[],{},existing,mode='deeper'),existing)
            measure.assert_not_called()

    def test_missing_camera_or_atlas_keeps_original(self):
        existing={(0,0,0)}
        self.assertEqual(model.recover(None,[],{},existing,mode='deeper'),existing)
        with patch('engine.section_relocation._atlas',return_value=None):
            self.assertEqual(model.recover(None,[{}],{'P':np.eye(3)},existing,mode='deeper'),existing)
