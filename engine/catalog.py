"""Versioned atlas and full-cube placement index derived from the installed client."""
import json
import gzip
from pathlib import Path
from zipfile import ZipFile

from calibration.models import Models, DIRECTIONS
from engine.atlas import make_atlas, discover
from engine.nbt import read_nbt


def template_format(archive):
    for path in sorted(archive.namelist()):
        if path.startswith(('data/minecraft/structure/','data/minecraft/structures/')) and path.endswith('.nbt'):
            data=read_nbt(gzip.decompress(archive.read(path)))
            palette=data.get('palette') or (data.get('palettes') or [[]])[0]
            if palette:
                for key in ('id','Name'):
                    if key in palette[0]:
                        return key
                raise ValueError(f'Unknown block state format in {path}')
    raise ValueError('Target client contains no readable vanilla structure template')


def build(version, out, jar=None, defaults=None):
    jar = Path(jar) if jar else discover(version)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atlas, _ = make_atlas(jar, out, version)
    with ZipFile(jar) as archive:
        release = json.loads(archive.read('version.json'))
        palette_key=template_format(archive)
        models = Models(archive)
        states = sorted(p for p in archive.namelist()
                        if p.startswith('assets/minecraft/blockstates/') and p.endswith('.json'))
        choices = {direction: {} for direction in DIRECTIONS}
        for path in states:
            block = Path(path).stem
            properties = (defaults or {}).get(block, {})
            for direction, lookup in choices.items():
                texture = models.front(block, properties, direction)
                if texture:
                    lookup.setdefault(texture, []).append(block)
    result = dict(version=version, data_version=release['world_version'],palette_key=palette_key,
                  pack_version=release['pack_version'], atlas_sha256=atlas['atlas_sha256'],
                  client_sha256=atlas['recipe']['client_sha256'], choices=choices)
    (out/'placement-index.json').write_text(json.dumps(result, separators=(',', ':')))
    return result
