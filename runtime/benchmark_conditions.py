"""Stage independent low-resolution and live warm-shader captures of the same texture panels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


def make_shader(jar, output):
    shader = 'assets/minecraft/shaders/core/terrain.fsh'
    line = 'fragColor = calculateFinalColor(color);'
    # Apply a mild cool shadow, warm highlights and different contrast inside
    # the live terrain program, before fog and before framebuffer capture.
    patch = """float lightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 grade = mix(vec3(0.91, 0.96, 1.06), vec3(1.16, 1.04, 0.82),
                     smoothstep(0.05, 0.70, lightness));
    color.rgb = clamp(pow(max(color.rgb, vec3(0.0)), vec3(1.08)) * grade, 0.0, 1.0);
    fragColor = calculateFinalColor(color);"""
    with ZipFile(jar) as client:
        text = client.read(shader).decode('utf-8')
        if text.count(line) != 1:
            raise ValueError('Unsupported terrain shader; inspect the installed release before applying the grade')
        meta = json.loads(client.read('version.json'))
    pack_version = meta['pack_version']
    version = [pack_version['resource_major'], pack_version['resource_minor']]
    with ZipFile(output, 'w', ZIP_DEFLATED) as pack:
        pack.writestr('pack.mcmeta', json.dumps({'pack': {'description': 'Live calibration terrain color grade',
            'min_format':version,'max_format':version}}))
        pack.writestr(shader, text.replace(line,patch))
    return hashlib.sha256(text.replace(line,patch).encode()).hexdigest()


def stage(source, jar):
    views = json.loads((source/'views.json').read_text())
    layout = json.loads((source/'layout.json').read_text())
    panels = [v for v in views if v.get('kind')=='adjacency']
    if not panels:
        raise ValueError('Missing adjacency panel views')
    pages = {v['page'] for v in panels}
    for name, resolution in [('small',[432,270]), ('warm-shader',[1440,900])]:
        out = source/'conditions'/name
        out.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source/'fixture',out/'fixture')
        (out/'views.json').write_text(json.dumps(panels))
        subset = dict(layout, blocks=[b for b in layout['blocks'] if b['page'] in pages])
        (out/'layout.json').write_text(json.dumps(subset))
        info = dict(condition=name,requested_resolution=resolution,views=len(panels),version=layout['version'])
        if name=='warm-shader':
            info['terrain_shader_sha256'] = make_shader(jar,out/'warm-terrain.zip')
        (out/'condition.json').write_text(json.dumps(info,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--jar',type=Path,required=True)
    args=p.parse_args()
    stage(args.source,args.jar)
