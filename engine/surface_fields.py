"""Single-frame CPU inference using the frozen local vision checkpoints."""
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from PIL import Image
from engine.atlas import digest
from engine.geometry import bilinear, project, transform


MODELS = {
    'normal': ('MINECRAFT_MOGE_ONNX','moge-2-vits.onnx','24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0'),
    'depth': ('MINECRAFT_DEPTH_ONNX','depth-v2-small-uint8.onnx','fcf51f1b230362b28690bb9d1809bf0431f29cad20534e3f589bd7285547f20d'),
}
SETTINGS = dict(provider='CPUExecutionProvider',normal_max_side=720,normal_tokens=1200,
                depth_size=518,threads=4,normal_resize='Lanczos',depth_resize='DPT aspect-preserving, bicubic')


def observed_normal(quad,normal,size):
    """Median of signed camera directions sampled away from the chart edges."""
    u,v=np.meshgrid(np.linspace(.1,.9,9),np.linspace(.1,.9,9))
    xy=project(transform(quad),np.stack([u.ravel(),v.ravel()],axis=1))
    vectors=bilinear(normal,xy*np.array(normal.shape[1::-1])/size)
    vec=np.median(vectors,axis=0)
    length=np.linalg.norm(vec)
    return (vec/length).tolist() if length>1.e-6 else None


@lru_cache(maxsize=2)
def session(kind,path):
    if not path.is_file() or digest(path)!=MODELS[kind][2]:
        raise ValueError(f'Provide the pinned {kind} model via {MODELS[kind][0]} or {path}')
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = SETTINGS['threads']
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,providers=[SETTINGS['provider']])


def infer(image,kind):
    env,filename,_ = MODELS[kind]
    path = Path(os.environ.get(env) or Path(__file__).resolve().parents[1]/'.deps'/filename)
    net = session(kind,path)
    image = image.convert('RGB')
    if kind=='normal':
        image.thumbnail((SETTINGS['normal_max_side'],)*2,Image.Resampling.LANCZOS)
        rgb = (np.asarray(image,dtype=np.float32)/255.).transpose(2,0,1)[None]
        output, = net.run(['normal'],dict(image=rgb,num_tokens=np.array(SETTINGS['normal_tokens'],dtype=np.int64)))
        if output.shape!=(1,image.height,image.width,3):
            raise ValueError(f'Unexpected normal map shape {output.shape}')
    else:
        from benchmark.depth_probe import preprocess
        output, = net.run(['predicted_depth'],{'pixel_values':preprocess(image)})
        if output.ndim!=3 or output.shape[0]!=1:
            raise ValueError(f'Unexpected inverse depth map shape {output.shape}')
    if not np.isfinite(output).all():
        raise ValueError(f'Nonfinite {kind} predictions')
    return output[0]
