import io
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from benchmark.count_export import _archive_pack, _parts
from benchmark.mask_blocks import _write_read, decoded_pack


class CountExport(unittest.TestCase):
    def test_chartless_and_partial_counts_preserve_regions_and_do_not_use_reference(self):
        regions = [dict(id=name, support={'runs': []}) for name in ('wall', 'thin')]
        quad = [[0, 0], [10, 0], [10, 10], [0, 10]]
        rows = [dict(id='wall', quad=quad, expected=[500, 500], measures={
            'masked-repeat': dict(counts=[6., 2.], phase=[None, None])}),
            dict(id='thin', quad=None, expected=[500, 500], measures={
                'masked-repeat': dict(counts=[2., None], phase=[.1, None])})]
        parts, phases = _parts(regions, rows, 'masked-repeat')
        self.assertEqual([p['pitch']['counts'] for p in parts], [[6., 2.], None])
        self.assertEqual([p['quad'] for p in parts], [quad, None])
        self.assertEqual([p['support'] for p in parts], [p['support'] for p in regions])
        self.assertEqual(phases, dict(resolved=0, partial=1, unresolved=1))
        with self.assertRaisesRegex(ValueError, 'identify the original'):
            _parts(regions, rows[::-1], 'masked-repeat')

    def test_archived_build_pack_decodes_to_the_original_blocks(self):
        cells = {(-35, -2, 8), (3, 4, 5)}
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)/'pack.zip'
            _write_read(src, cells)
            outer = io.BytesIO()
            with ZipFile(outer, 'w') as archive:
                _archive_pack(archive, 'a/b.zip', src)
            with ZipFile(outer) as archive:
                payload = archive.read('a/b.zip')
            self.assertEqual(decoded_pack(io.BytesIO(payload)), {p:'minecraft:smooth_stone' for p in cells})
            with ZipFile(io.BytesIO(payload)) as packed:
                self.assertTrue(all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in packed.infolist()))
