"""Regression cases for visibility and one-to-one cell membership."""
import unittest
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from benchmark.textures import clear_rays, match_cells
from engine.geometry import UNIT, project


class TextureMembershipTests(unittest.TestCase):
    def test_build_cli_opt_in_exposure(self):
        cli=Path(__file__).resolve().parents[1]/'reconstruct.py'
        rng=np.random.default_rng(88)
        bright=rng.integers(148,205,(16,16))
        dark=rng.integers(60,94,(16,16))*.6/.44
        def rgb(a):
            return np.repeat(np.uint8(a)[:,:,None],3,axis=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tiles=np.stack([rgb(bright),rgb(dark)])
            Image.fromarray(np.concatenate(tiles,axis=1)).save(root/'atlas.png')
            stamp=hashlib.sha256((root/'atlas.png').read_bytes()).hexdigest()
            meta=dict(version='test',atlas_sha256=stamp)
            (root/'atlas.json').write_text(json.dumps(dict(meta,atlas='atlas.png')))
            labels=['stone','deepslate']
            catalog=dict(meta,choices={'south':{name:[name] for name in labels}},data_version=5023,
                         palette_key='id',pack_version={'data_major':121,'data_minor':0})
            (root/'placement-index.json').write_text(json.dumps(catalog))
            np.savez(root/'library.npz',tiles=tiles,labels=labels,method='multiscale',**meta)
            Image.fromarray(rgb(bright*.6)).save(root/'wall.png')
            scene=dict(planes=[dict(name='wall',columns=1,rows=1,quad=[[0,0],[16,0],[16,16],[0,16]])])
            (root/'scene.json').write_text(json.dumps(scene))
            for flag,expected in [(False,'deepslate'),(True,'stone')]:
                cmd=[sys.executable,str(cli),'build',str(root/'wall.png'),'--atlas',str(root),
                     '--out',str(root/'out'),'--library',str(root/'library.npz'),'--scene',str(root/'scene.json')]
                if flag: cmd.append('--match-exposure')
                subprocess.run(cmd,check=True,capture_output=True)
                result=json.loads((root/'out/matches.json').read_text())
                self.assertEqual(result['match_exposure'],flag)
                self.assertEqual(result['cells'][0]['block_id'],expected)

    def test_one_to_one_perspective(self):
        h=np.array([[8.,1.,80.],[-1.,9.,60.],[.01,-.005,1.]])
        # One oversized proposal and two competing proposals for the first cell.
        crops=[UNIT*[3,1],UNIT+.04,UNIT,UNIT+[1,0]]
        recovered,_=match_cells([project(h,q) for q in crops],{(0,0),(1,0),(2,0)},np.linalg.inv(h))
        self.assertEqual(set(recovered),{(0,0),(1,0)})
        self.assertAlmostEqual(recovered[0,0][1],1.)
        self.assertAlmostEqual(recovered[1,0][2],1.)

    def test_overhang_blocks_ray(self):
        cells={(0,0,0),(0,1,1),(1,0,0)}
        starts=np.array([[.5,.9,1.00001],[1.5,.9,1.00001]])
        visible=clear_rays(cells,np.array([1.5,4.,5.]),starts)
        self.assertEqual(visible.tolist(),[False,True])
