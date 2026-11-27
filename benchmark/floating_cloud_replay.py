"""Artifact roots and provenance for the active floating-build benchmark."""

from benchmark.expanded_baseline import ROOT, digest, read


RUN = ROOT / 'runs/floating-cloud-gate-01'
SHARED_ROWS = RUN / 'shared-corner-rows'
NATIVE_COUNTER = RUN / 'surface-crop-first'
ANCHOR_EXTENSION = RUN / 'garden-anchor-extension'
ASSEMBLY = RUN / 'relative-face-assembly'


def source_handoff(name):
    return RUN / 'sky' / name / 'source-handoff'


def handoff_provenance(handoff, source_sha256):
    manifest = handoff / 'build-manifest.json'
    metadata = read(manifest)
    method = metadata['stages']['background_filter']
    if method != 'border-connected-whole-sky-v1' or source_sha256 != metadata['image_sha256']:
        raise ValueError(f'Expected filtered handoff for the same screenshot: {handoff}')
    return dict(background_filter=method, source_handoff_sha256=digest(manifest),
                surfaces_sha256=digest(handoff / 'surfaces.json'),
                counts_sha256=digest(handoff / 'counts.json'))


def check_handoff(packet, handoff, source_sha256):
    expected = handoff_provenance(handoff, source_sha256)
    if any(packet.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Chart packet differs from the active filtered handoff: {handoff}')
    return expected
