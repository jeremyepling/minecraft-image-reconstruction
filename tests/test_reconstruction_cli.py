"""Check the public command surface separately from the historical benchmark API."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import reconstruct


class ReconstructionCliTests(unittest.TestCase):
    def test_counts_use_project_defaults_and_forward_crop_and_atlas(self):
        argv = ['reconstruct', 'build', 'picture.png', '--through', 'counts', '--version', '26.3',
                '--surface-context-px', '10', '--material-match', 'atlas', '--atlas', 'atlas', '--out', 'out']
        with patch.object(sys, 'argv', argv), patch('engine.surface_model.build') as build:
            reconstruct.main()
        build.assert_called_once_with(Path('picture.png'), '26.3', Path('out'), through='counts',
                                      atlas_path=Path('atlas'), material_match='atlas', context_px=10)

    def test_prototype_commands_and_method_selectors_are_retired(self):
        retired = [[cmd] for cmd in ('inspect-openings', 'inspect-form', 'fit-voxels')]
        for flag, value in [('--surface-outlines', 'mask-first'), ('--surface-boundary', 'wall-recovery'),
                            ('--surface-order', 'geometry'), ('--block-counter', 'masked-repeat'),
                            ('--geometry-method', 'cube')]:
            retired.append(['build', 'picture.png', '--through', 'counts', '--version', '26.3',
                            '--out', 'out', flag, value])
        retired.append(['analyze', 'picture.png', '--out', 'out', '--geometry-method', 'texel'])
        for args in retired:
            with self.subTest(args=args), patch.object(sys, 'argv', ['reconstruct', *args]), redirect_stderr(StringIO()) as err:
                with self.assertRaises(SystemExit) as rejected:
                    reconstruct.main()
                self.assertEqual(rejected.exception.code, 2)
                if len(args) == 1:
                    self.assertIn('invalid choice', err.getvalue())
                else:
                    self.assertIn(f'unrecognized arguments: {args[-2]}', err.getvalue())
