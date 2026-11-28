"""Source-image normal and depth fields used by the Floating predictor."""
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from engine import surface_fields, surface_model, visible_placement


def verify_fields(directory, source_sha):
    """Check the image, both pinned checkpoints, preprocessing, and field bytes."""
    receipt = directory / 'receipt.json'
    proof = read(receipt)
    models = {kind: spec[2] for kind, spec in surface_fields.MODELS.items()}
    if (proof.get('source_sha256') != source_sha or proof.get('model_sha256') != models
            or proof.get('preprocessing') != surface_fields.SETTINGS
            or any(digest(directory / (kind + '.npz')) != proof.get('fields_sha256', {}).get(kind)
                   for kind in models)):
        raise ValueError(f'Invalid frozen model field receipt: {receipt}')
    return directory


def infer_fields(source, source_sha, directory):
    directory.mkdir(parents=True)
    models = {kind: spec[2] for kind, spec in surface_fields.MODELS.items()}
    hashes = {}
    with Image.open(source) as image:
        for kind in models:
            path = directory / (kind + '.npz')
            np.savez_compressed(path, value=surface_fields.infer(image, kind))
            hashes[kind] = digest(path)
    save(directory / 'receipt.json', dict(source_sha256=source_sha, model_sha256=models,
                                          fields_sha256=hashes, preprocessing=surface_fields.SETTINGS))
    return directory


def private_fields(source, source_sha, archived, private, frame_id=None):
    """Locate fields by source digest or verified legacy receipt, or infer privately."""
    if digest(source) != source_sha:
        raise ValueError(f'Source hash mismatch: {source}')
    directory = archived / source_sha
    if (directory / 'receipt.json').is_file():
        return verify_fields(directory, source_sha)
    if frame_id:
        for donor in (archived / frame_id, archived / 'original' / frame_id / 'fields'):
            receipt = donor / 'receipt.json'
            if receipt.is_file() and read(receipt).get('source_sha256') == source_sha:
                return verify_fields(donor, source_sha)
    return infer_fields(source, source_sha, private)


def build_handoff(source, cache, handoff):
    from engine.floating_sky import METHOD
    receipt = handoff / 'build-manifest.json'
    if receipt.exists():
        saved = read(receipt)
        if saved['image_sha256'] == digest(source) and saved['stages'].get('background_filter') == METHOD:
            return

    def field(image, kind):
        with np.load(cache / (kind + '.npz')) as data:
            return data['value']

    with patch.object(surface_fields, 'infer', field), patch.object(
            visible_placement, 'seam_camera', side_effect=AssertionError("No floor-camera calls permitted")):
        surface_model.build(source, '26.3', handoff, through='counts', boundary='wall-depth-anything')
