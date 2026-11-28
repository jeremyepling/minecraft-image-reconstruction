"""Build a Java 26.3 datapack from one screenshot with Expanded Split-Border."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from engine.floating_fields import build_handoff, infer_fields, verify_fields
from runtime.mode import image_only


METHOD = 'expanded-split-border'
# The archived chart producers and join solver bind paths/functions at module scope.
# Distinct processes can predict concurrently; keep their overrides local to one call per process.
_BUILD_LOCK = Lock()


def _publish(outputs, out, scratch):
    """Replace the products together and restore the old set on a publication error."""
    backup = scratch / 'previous'
    backup.mkdir()
    with ExitStack() as restore:
        for name, source in outputs.items():
            target = out / name
            if target.exists():
                old = backup / name
                target.replace(old)
                restore.callback(old.replace, target)
            source.replace(target)
            restore.callback(target.replace, source)
        restore.pop_all()


def build(source: Path, out: Path, *, field_cache: Path | None = None):
    """Return the selected assembly and write its verified datapack and source evidence.

    A supplied cache is the directory containing normal.npz, depth.npz, and receipt.json.
    Otherwise infer both fields and publish them under out/fields.
    """
    source = Path(source).resolve()
    out = Path(out).resolve()
    source_sha = digest(source)
    out.mkdir(parents=True, exist_ok=True)
    with _BUILD_LOCK, image_only(), TemporaryDirectory(dir=out, prefix='.floating-') as work:
        from benchmark import floating_v1_expanded as expanded, floating_v1_run as v1
        from benchmark.floating_selective_replay import assemble_arm

        scratch = Path(work)
        outputs = {}
        if field_cache is None:
            cache = infer_fields(source, source_sha, scratch / 'fields')
            outputs['fields'] = cache
        else:
            cache = verify_fields(Path(field_cache), source_sha)

        # Archive layout uses a frame key and variant for addressing its chart packets.
        # Keep them constant for every input, independent of the screenshot filename.
        location = Path('sky/input')
        case = dict(id='input', rgb_sha256=source_sha,
                    variants=dict(sky=dict(image=str(source), source_sha256=source_sha)))
        handoff = scratch / location / 'source-handoff'
        build_handoff(source, cache, handoff)
        with patch.object(v1, 'SOURCE', scratch), patch.object(v1, 'OUT', scratch):
            v1.native_charts(case)
            _, decisions = expanded.expanded_packet(case, scratch, field_cache=cache)
        charts = scratch / 'split-border/charts' / location / 'charts.json'
        packet = read(charts)
        v1.paths.check_handoff(packet, handoff, source_sha)
        planes = read(handoff / 'surfaces.json')['planes']
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB'))
        destination = scratch / 'assembly'
        record = assemble_arm(packet, planes, rgb, {}, 2.5, destination, save_trace=True)
        save(scratch / 'decisions.json', dict(method=METHOD, source_sha256=source_sha, **decisions))
        outputs.update({'source-handoff': handoff, 'charts.json': charts,
                        'decisions.json': scratch / 'decisions.json',
                        'assembly.json': destination / 'assembly.json',
                        'oriented-charts.json': destination / 'oriented-charts.json',
                        'reconstruction.zip': destination / 'reconstruction.zip'})
        _publish(outputs, out, scratch)
        return record
