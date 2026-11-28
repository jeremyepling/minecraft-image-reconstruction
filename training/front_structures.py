"""Save loadable Java 26.3 structures and the native recipes used for a Front 100 release."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from calibration.world import PACK, tag
from engine.atlas import digest, resource_id
from engine.nbt import read_nbt
from training.decorative_labels import properties_for


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def member(archive, name, data):
    info = ZipInfo(name)
    info.compress_type = ZIP_DEFLATED
    archive.writestr(info, data)


def template(readback, data_version):
    cells = {}
    for row in readback:
        name = resource_id(row['block_id'])
        if name == 'minecraft:air':
            continue
        point = tuple(row[axis] for axis in 'xyz')
        if point in cells:
            raise ValueError(f'Duplicate source cell {point}')
        cells[point] = (name, tuple(sorted(properties_for(row).items())))
    origin = [min(p[i] for p in cells) for i in range(3)]
    size = [1 + max(p[i] for p in cells) - origin[i] for i in range(3)]
    states = sorted(set(cells.values()))
    # Java 26.3 serializes palette block states as lowercase id/properties.
    palette = [dict(id=name, **({'properties': dict(props)} if props else {})) for name, props in states]
    indices = {state: i for i, state in enumerate(states)}
    blocks = [dict(pos=[v-o for v, o in zip(point, origin)], state=indices[cells[point]])
              for point in sorted(cells)]
    nbt = dict(size=size, blocks=blocks, palette=palette, entities=[], DataVersion=data_version)
    compressed = gzip.compress(b'\x0a\x00\x00' + tag(nbt)[1], mtime=0)
    saved = read_nbt(gzip.decompress(compressed))
    decoded = {}
    for block in saved['blocks']:
        pos = tuple(v+o for v, o in zip(block['pos'], origin))
        state = saved['palette'][block['state']]
        decoded[pos] = (state['id'], tuple(sorted(state.get('properties', {}).items())))
    if decoded != cells or saved['size'] != size or saved['DataVersion'] != data_version:
        raise ValueError('Decoded template differs from the native world readback')
    return compressed, dict(origin_world=origin, size_xyz=size, blocks=len(cells), states=len(states),
                            data_version=data_version, sha256=sha256(compressed).hexdigest())


def build_archive(capture_roots, output_path):
    """Create one installable datapack containing both versions of each build and their source records."""
    output_path = Path(output_path)
    mappings = {}
    records = {}
    with ZipFile(output_path, 'x') as archive:
        for batch, root in enumerate(sorted(map(Path, capture_roots)), 1):
            prefix = f'capture/batch-{batch:02d}/'
            world = root / 'fixture/saves/Palette Lab'
            level = read_nbt(gzip.decompress((world / 'level.dat').read_bytes()))
            version = level['Data']['DataVersion']
            with ZipFile(world / 'datapacks' / PACK) as source:
                mcmeta = source.read('pack.mcmeta')
            if batch == 1:
                metadata = json.loads(mcmeta)
                metadata['pack']['description'] = 'Floating Front 100 native structure templates'
                member(archive, 'pack.mcmeta', encode(metadata))
            paths = [root / 'layout.json', root / 'views.json', root / 'client/options.txt',
                     root / 'runtime-block-registry.json',
                     world / 'level.dat', world / 'datapacks' / PACK, *sorted((world / 'data').rglob('*.dat'))]
            for path in paths:
                name = prefix + path.relative_to(root).as_posix()
                payload = path.read_bytes()
                member(archive, name, payload)
                records[name] = sha256(payload).hexdigest()
            renderer = [line.split(']: ', 1)[-1] for line in (root / 'client-run.log').read_text(
                encoding='utf-8', errors='replace').splitlines()
                        if any(s in line for s in ('Backend library:', 'Using graphics backend', 'OpenGL Renderer:'))]
            member(archive, prefix + 'renderer.json', encode(renderer))
            if batch == 1:
                for filename in ('capture-run.json', 'capture-environment.json'):
                    receipt = root.parent / filename
                    if receipt.is_file():
                        member(archive, 'capture/' + filename, receipt.read_bytes())
            for path in sorted((root / 'frames').glob('*/frame.json')):
                frame = json.loads(path.read_bytes())
                if frame['lighting'] != 'noon':
                    continue
                key = f"{frame['layout']}/{frame['condition']}"
                if key in mappings:
                    raise ValueError(f'Duplicate structure {key}')
                data = (path.parent / 'world-readback.json').read_bytes()
                if sha256(data).hexdigest() != frame['files']['world-readback.json']:
                    raise ValueError(f"World readback changed for {frame['id']}")
                nbt, summary = template(json.loads(data), version)
                name = 'data/floating_front/structure/' + key + '.nbt'
                member(archive, name, nbt)
                mappings[key] = dict(summary, source_frame=frame['id'], path=name,
                                     world_readback_sha256=frame['files']['world-readback.json'],
                                     capture_records=prefix, resource='floating_front:' + key)
        layouts = {key.split('/')[0] for key in mappings}
        if not layouts or any(f'{layout}/{condition}' not in mappings for layout in layouts
                              for condition in ('full', 'decorative')):
            raise ValueError('Expected full and decorative templates for every captured layout')
        member(archive, 'structures.json', encode(dict(schema=1, templates=mappings, capture_files=records,
               placement='Place in empty space at origin_world to restore the recorded world positions.')))
    return dict(file=output_path.name, sha256=digest(output_path), bytes=output_path.stat().st_size,
                layouts=len(layouts), templates=len(mappings))
