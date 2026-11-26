"""Emit sparse, tiled structure templates and a placement function for the selected release."""
import gzip
import json
from collections import defaultdict
from zipfile import ZipFile, ZIP_DEFLATED

from calibration.world import tag


# Verified against the 26.3 release metadata and vanilla structure palette.
JAVA_26_3_FORMAT = dict(data_version=5023, palette_key='id', pack_version=dict(data_major=121, data_minor=0))


def write_pack(path, blocks, meta):
    pack=meta['pack_version']
    major=pack['data_major'] if 'data_major' in pack else pack['data']
    desc={'description':'Reconstructed Minecraft build'}
    if major>=88:
        ver=[major,pack.get('data_minor',0)]
        desc.update(min_format=ver,max_format=ver)
    else:
        desc['pack_format']=major
    singular=major>=45
    template_dir='structure' if singular else 'structures'
    function_dir='function' if singular else 'functions'
    chunks=defaultdict(list)
    for pos,name in sorted(blocks.items()):
        chunks[tuple(n//32 for n in pos)].append((pos,name))
    with ZipFile(path,'w',ZIP_DEFLATED) as out:
        out.writestr('pack.mcmeta',json.dumps({'pack':desc}))
        commands=[]
        for i,group in enumerate(chunks.values()):
            low=tuple(min(p[a] for p,_ in group) for a in range(3))
            size=[max(p[a] for p,_ in group)-low[a]+1 for a in range(3)]
            names=sorted({name for _,name in group})
            indices={name:i for i,name in enumerate(names)}
            data=dict(size=size,palette=[{meta['palette_key']:'minecraft:'+s} for s in names],
                blocks=[dict(pos=[a-b for a,b in zip(p,low)],state=indices[s]) for p,s in group],
                entities=[],DataVersion=meta['data_version'])
            payload=gzip.compress(bytes([10,0,0])+tag(data)[1],mtime=0)
            out.writestr(f'data/reconstruction/{template_dir}/part_{i}.nbt',payload)
            offset=' '.join(f'~{v}' for v in low)
            commands.append(f'place template reconstruction:part_{i} {offset}')
        out.writestr(f'data/reconstruction/{function_dir}/build.mcfunction','\n'.join(commands)+'\n')
