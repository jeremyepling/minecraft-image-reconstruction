"""Exercise the public scene and saved-grid material paths with a small, known atlas."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from engine.atlas import digest
from engine.masonry import competing_face, pattern_templates
from engine.reconstruction import reconstruct


class ContextBuildTest(unittest.TestCase):
    def test_staggered_planks_can_veto_a_brick_proposal_after_a_grid_shift(self):
        y, x = np.mgrid[:16, :16]
        joints = (y % 8 == 0) | ((x + 8*(y//8)) % 16 == 0)
        mortar = np.repeat((155 - 60*joints).astype('uint8')[:, :, None], 3, axis=2)
        grain = ((3*x + 5*(y//2)) % 7) + 24*((y % 4) == 0)
        plank = (np.array([152, 104, 78]) - grain[:, :, None]).astype('uint8')
        bank = pattern_templates(np.stack([mortar, plank]), np.array(['stone_bricks','oak_planks']))
        guess = competing_face(np.roll(plank, (3, 7), axis=(0, 1)), bank)
        self.assertEqual(guess[0][0], 'oak_planks')
        self.assertIsNone(competing_face(np.roll(mortar, (3, 7), axis=(0, 1)), bank))
        self.assertIsNone(competing_face(np.full_like(plank, 160), bank))

    def test_shifted_warm_brick_beside_smooth_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            y, x = np.mgrid[:16, :16]
            mortar = (y % 8 == 0) | ((x + 8*(y//8)) % 16 == 0)
            brick = np.repeat((165 - 55*mortar).astype('uint8')[:, :, None], 3, axis=2)
            pink = np.full_like(brick, [177, 159, 151])
            tiles = np.stack([brick, pink])
            names = ['stone_bricks', 'white_terracotta']
            sheet = root/'atlas.png'
            Image.fromarray(np.concatenate(tiles, axis=1)).save(sheet)
            common = dict(version='test', atlas_sha256=digest(sheet))
            textures = [dict(id=name, rect=[i*16, 0, 16, 16]) for i, name in enumerate(names)]
            (root/'atlas.json').write_text(json.dumps(dict(common, atlas=sheet.name, textures=textures)))
            index = dict(common, choices={'south': {n: [n] for n in names}}, pack_version={'data': 90},
                         data_version=5000, palette_key='Name')
            (root/'placement-index.json').write_text(json.dumps(index))
            library = root/'library.npz'
            np.savez(library, **common, tiles=tiles, labels=names, method='multiscale', task='face_textures')

            image = np.tile(brick, (6, 8, 1))
            image = np.roll(image, (3, 5), axis=(0, 1))
            image = (image.astype(int) + [13, -3, -9]).astype('uint8')
            image[:, 80:] = pink[0, 0]
            source = root/'wall.png'
            Image.fromarray(image).save(source)
            scene = root/'scene.json'
            scene.write_text(json.dumps(dict(planes=[dict(name='wall', columns=8, rows=6,
                quad=[[0,0],[128,0],[128,96],[0,96]], origin=[0,0,0], facing='south')])))

            first = root/'guided'
            reconstruct(source, root, scene, library, first, match_appearance='context')
            def read(folder, name):
                return json.loads((folder/name).read_text())
            matches = read(first, 'matches.json')
            self.assertEqual(matches['method'], 'appearance-context')
            by_cell = {tuple(c['ij']): c for c in matches['cells']}
            pier = by_cell[2, 3]
            self.assertEqual((pier['material_method'], pier['block_id']), ('masonry', 'stone_bricks'))
            self.assertGreater(pier['masonry_evidence']['atlas_correlation'], .18)
            self.assertEqual(by_cell[6, 3]['block_id'], 'white_terracotta')
            self.assertNotIn('masonry_evidence', by_cell[6, 3])
            grid = first/'geometry.json'
            replay = root/'replay'
            reconstruct(source, root, None, library, replay, grid=grid, match_appearance='context')
            self.assertEqual(read(first,'placements.json'), read(replay,'placements.json'))
            self.assertEqual(matches, read(replay,'matches.json'))

            reconstruct(source, root, scene, library, replay, match_appearance='shared')
            self.assertEqual(read(replay,'matches.json')['method'], 'appearance-shared')


if __name__ == '__main__':
    unittest.main()
