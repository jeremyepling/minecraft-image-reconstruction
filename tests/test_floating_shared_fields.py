"""Model field caches are immutable inputs shared by both active Floating cohorts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.floating_build_local_run import private_fields
from benchmark.expanded_baseline import digest, read
from engine import surface_fields


class FloatingSharedFieldTest(unittest.TestCase):
    def test_private_inference_and_read_only_reuse(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.png'
            Image.new('RGB', (4, 4)).save(source)
            sha = digest(source)
            shared = root/'shared'
            private = root/'first-run'/'fields'
            with patch.object(surface_fields, 'infer', return_value=np.zeros((4, 4))) as infer:
                self.assertEqual(private_fields(source, sha, shared, private), private)
            self.assertEqual(infer.call_count, 2)
            self.assertFalse(shared.exists())
            self.assertEqual(read(private/'receipt.json')['source_sha256'], sha)

            shared.mkdir()
            captured = private.rename(shared/sha)
            receipt_sha = digest(captured/'receipt.json')
            second = root/'second-run'/'fields'
            with patch.object(surface_fields, 'infer', side_effect=AssertionError):
                self.assertEqual(private_fields(source, sha, shared, second), captured)
                self.assertFalse(second.exists())
                self.assertEqual(digest(captured/'receipt.json'), receipt_sha)
                donor = captured.rename(shared/'capture')
                self.assertEqual(private_fields(source, sha, shared, second, 'capture'), donor)
                (donor/'depth.npz').write_bytes(b'changed cache')
                with self.assertRaisesRegex(ValueError, 'Invalid frozen model field'):
                    private_fields(source, sha, shared, second, 'capture')

    def test_source_mismatch_fails_before_writing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.png'
            source.write_bytes(b'wrong image')
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                private_fields(source, '0'*64, root/'shared', root/'private')
            self.assertFalse((root/'private').exists())
