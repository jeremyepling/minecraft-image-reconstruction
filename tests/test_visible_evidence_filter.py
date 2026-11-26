import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import visible_evidence_filter


class VisibleEvidenceFilterTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (96, 96), 'gray')
        K = np.array([[75, 0, 48], [0, 75, 48], [0, 0, 1]])
        axes = np.diag([1, -1, 1])
        self.P = K @ np.column_stack((axes, -axes @ [.5, .5, -5]))
        self.cells = {(0, 0, 0), (0, 0, 1)}

    def test_verified_weights_and_fallbacks(self):
        verify=visible_evidence_filter.filter_cells
        parts=[dict(normal_camera=[0.,0.,-1.],support={'size':[24,24],'runs':[[i,0,24] for i in range(24)]})]
        self.assertEqual(len(visible_evidence_filter._model_paths()),2)
        for projection in (None,[[1,2],[3]],np.full((3,4),np.nan),np.zeros((3,4))):
            self.assertIs(verify(self.image,parts,{'P':projection},self.cells),self.cells)
        self.assertIs(verify(self.image,[],{'P':self.P},self.cells),self.cells)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'model.json'
            with patch.object(visible_evidence_filter,'MODEL',path):
                for content in (None,'[bad','{"paths":[]}'):
                    if content is not None:
                        path.write_text(content)
                    self.assertIs(verify(self.image,parts,{'P':self.P},self.cells),self.cells)
            self.assertIs(verify(self.image,parts,{'P':self.P},self.cells,Path(folder)),self.cells)
