"""Regression cases for release-specific NBT, portable placement and shared geometry."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image
from calibration.world import tag
from engine.catalog import template_format
from engine.export import write_pack
from engine.nbt import read_nbt
from engine.reconstruction import analyze


class ReconstructionTests(unittest.TestCase):
    def test_unannotated_independent_wall(self):
        image=Path(__file__).resolve().parents[1]/'examples/repair-wall/wall-only.png'
        with tempfile.TemporaryDirectory() as tmp:
            _,planes,cells=analyze(image,None,Path(tmp))
            self.assertEqual([(p['columns'],p['rows']) for p in planes],[(6,6)])
            self.assertEqual(len({tuple(c['xyz']) for c in cells}),36)

    def test_template_release_round_trip(self):
        blocks={(-1,0,0):'tuff',(32,0,0):'stone'}
        for key in ('Name','id'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                native=root/'native.jar'
                raw={'palette':[{key:'minecraft:stone'}]}
                with ZipFile(native,'w') as z:
                    z.writestr('data/minecraft/structure/test.nbt',gzip.compress(bytes([10,0,0])+tag(raw)[1]))
                with ZipFile(native) as z:
                    detected=template_format(z)
                self.assertEqual(key,detected)
                result=root/'out.zip'
                write_pack(result,blocks,dict(data_version=5000,palette_key=detected,pack_version={'data_major':121,'data_minor':0}))
                with ZipFile(result) as z:
                    function=z.read('data/reconstruction/function/build.mcfunction').decode()
                    self.assertIn('~-1 ~0 ~0',function)
                    self.assertIn('~32 ~0 ~0',function)
                    for i,expected in enumerate(['tuff','stone']):
                        data=read_nbt(gzip.decompress(z.read(f'data/reconstruction/structure/part_{i}.nbt')))
                        self.assertEqual(data['palette'],[{key:'minecraft:'+expected}])
                        self.assertEqual(data['blocks'][0]['pos'],[0,0,0])

    def test_roof_orientation_and_openings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); img=root/'test.png'; scene=root/'scene.json'
            Image.new('RGB',(48,32),'gray').save(img)
            scene.write_text(json.dumps(dict(planes=[dict(name='roof',quad=[[0,0],[48,0],[48,32],[0,32]],
                columns=3,rows=2,facing='up',origin=[6,9,3],holes=[[[0,0],[16,0],[16,16],[0,16]]])])))
            _,planes,cells=analyze(img,scene,root)
            self.assertEqual(len(cells),5)
            self.assertEqual(cells[-1]['xyz'],[8,9,3])
            self.assertEqual(cells[0]['xyz'],[7,9,2])
            dim=json.loads((root/'dimensions.json').read_text())
            self.assertEqual(dim['observed_size_xyz'],[3,1,2])


if __name__=='__main__':
    unittest.main()
