"""Verify frame publication, candidate isolation and capture failure propagation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

from benchmark import planes
from benchmark.geometry_benchmark import load_prediction
from runtime.capture import run_client


class CapturePredictionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.frames = self.root / 'captures'
        self.frames.mkdir()
        self.predictions = self.root / 'candidate'
        views = [dict(id=name, kind='canopy', split='dev') for name in ('front', 'back')]
        (self.root / 'views.json').write_text(json.dumps(views))

    def predict(self, active=None):
        planes.run(self.root, ['dev'], 1, 0, ['partition'], self.predictions, active)

    def test_predictions_stream_before_capture_exits_and_replay_reuses_cache(self):
        Image.new('RGB', (64, 64), 'gray').save(self.frames / 'front.png')
        publisher = """
from pathlib import Path
from time import sleep
from shutil import copyfile
frames = Path('captures')
(frames / 'front.json').write_text('{}')
while not Path('candidate/partition/front.json').is_file():
    sleep(0.05)
copyfile(frames / 'front.png', frames / 'back.png')
(frames / 'back.json').write_text('{}')
"""
        run_client([sys.executable, '-c', publisher], self.root,
                   self.root / 'producer.log', 30, self.predict)
        paths = sorted((self.predictions / 'partition').glob('*.json'))
        self.assertEqual([p.stem for p in paths], ['back', 'front'])
        self.assertFalse((self.root / 'geometry-predictions').exists())
        digest = hashlib.sha256((self.frames / 'front.png').read_bytes()).hexdigest()
        self.assertTrue(all(json.loads(p.read_text())['image_hash'] == digest for p in paths))
        with self.assertRaisesRegex(ValueError, 'different image: front'):
            load_prediction(self.predictions, 'partition', 'front', 'other-fixture')
        times = [p.stat().st_mtime_ns for p in paths]
        self.predict()
        self.assertEqual(times, [p.stat().st_mtime_ns for p in paths])

    def test_successful_producer_with_unfinished_frames_fails(self):
        # A PNG without its pose and a pose without its PNG are both incomplete.
        (self.frames / 'front.png').touch()
        (self.frames / 'back.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, '2 expected screenshots are missing'):
            run_client([sys.executable, '-c', 'pass'], self.root,
                       self.root / 'producer.log', 30, self.predict)
        self.assertFalse(self.predictions.exists())

    def test_client_failure_reaches_caller(self):
        with self.assertRaisesRegex(RuntimeError, 'status 7'):
            run_client([sys.executable, '-c', 'raise SystemExit(7)'], self.root,
                       self.root / 'producer.log', 30, self.predict)
