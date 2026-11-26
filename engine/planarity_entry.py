"""Single-screenshot numerical evidence for Plane Recovery, in a separate model process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from engine.atlas import digest
from engine.planarity_source import MODEL_REV, MODEL_SHA, MOGE, SETTINGS, UPSTREAM, UTILS3D, infer_image, load_model
from engine.recovery_evidence import ROOT, _model_input


LABELS = 'relaxed'
PRESET = 'relaxed-add-split'
METHOD = 'plane-recovery'
RELEASE = dict(upstream=UPSTREAM, moge=MOGE, utils3d=UTILS3D, model_revision=MODEL_REV,
               checkpoint_sha256=MODEL_SHA)


def selected_settings(settings):
    return {**{k:settings[k] for k in SETTINGS if k!='segmentations'},
            'normal_degrees':settings['segmentations'][LABELS][0],
            'minimum_neighbors':settings['segmentations'][LABELS][1]}


def infer(source, device):
    """Use a content-addressed cache only when explicitly supplied. Unknown RGB is inferred."""
    source = Path(source).resolve()
    sha = digest(source)
    cache = os.environ.get('MCIR_PLANARITY_FIELDS')
    if cache:
        folder = Path(cache)
        manifest = json.loads((folder/'manifest.json').read_bytes())
        if (any(manifest.get(k)!=v for k,v in RELEASE.items()) or
                selected_settings(manifest['settings'])!=selected_settings(SETTINGS)):
            raise ValueError('Pixel-wise Planarity cache uses a different release or settings')
        name = f'fields/{sha}.npz'
        if name in manifest['artifacts']:
            path = folder/name
            if digest(path)!=manifest['artifacts'][name]:
                raise ValueError('Pixel-wise Planarity cached field checksum differs')
            with np.load(path,allow_pickle=False) as numeric:
                fields = {k:numeric[k] for k in (LABELS,LABELS+'_normals',LABELS+'_centers','image_size')}
            return fields,dict(RELEASE,source='verified numerical cache',image_sha256=sha,
                               settings=selected_settings(SETTINGS),field_sha256=manifest['artifacts'][name],
                               cache_manifest_sha256=digest(folder/'manifest.json'),inference_s=0)
    interpreter = os.environ.get('MCIR_RECOVERY_PYTHON') or sys.executable
    environment = os.environ.copy()
    extra = environment.get('MCIR_PLANARITY_PYTHONPATH')
    if extra:
        environment['PYTHONPATH'] = os.pathsep.join(filter(None,(extra,environment.get('PYTHONPATH'))))
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        subprocess.run([interpreter,'-B','-m','engine.planarity_entry',str(source),str(folder),device],
                       cwd=ROOT,env=environment,check=True)
        with np.load(folder/'fields.npz',allow_pickle=False) as numeric:
            fields = dict(numeric)
        info = json.loads((folder/'model.json').read_bytes())
    return fields,info


def compute(source,folder,device):
    import torch
    upstream = _model_input('MCIR_PLANARITY_SOURCE','PixelwisePlanarity')
    utils3d = _model_input('MCIR_PLANARITY_UTILS3D','utils3d')
    checkpoint = _model_input('MCIR_PLANARITY_CHECKPOINT','pixelwise-planarity.pt')
    model,segmentation,load_s = load_model(upstream,utils3d,checkpoint,device)
    with Image.open(source) as photo:
        fields,timing = infer_image(photo.convert('RGB'),model,segmentation,device,[LABELS])
    np.savez_compressed(folder/'fields.npz',**fields)
    info = dict(RELEASE,source='fresh screenshot inference',image_sha256=digest(source),device=device,
                torch_version=torch.__version__,settings=selected_settings(SETTINGS),load_s=load_s,
                field_sha256=digest(folder/'fields.npz'),seed=0,**timing)
    (folder/'model.json').write_text(json.dumps(info,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('folder',type=Path)
    parser.add_argument('device',choices=('cpu','cuda'))
    args = parser.parse_args()
    compute(args.source,args.folder,args.device)
