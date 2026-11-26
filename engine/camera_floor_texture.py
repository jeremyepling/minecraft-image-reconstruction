"""Image-only floor-texture cameras. Earlier rounds are retained for paired replay."""
import numpy as np
from PIL import Image
from engine import camera_seam


def normalized_floor(image):
    rgb = np.asarray(image.convert('RGB'),float)
    h,w=rgb.shape[:2]
    sample=rgb[int(h*.82):int(h*.97),:,:]
    center=np.median(sample,axis=(0,1))
    dynamic=np.percentile(sample,90,axis=(0,1))-np.percentile(sample,10,axis=(0,1))
    return Image.fromarray(np.uint8(np.clip(128+(rgb-center)*np.minimum(40.,60/np.maximum(dynamic,1)),0,255)))


def predict(image, version='26.3'):
    candidate=camera_seam.predict(normalized_floor(image),version)
    candidate['floor_method']='contrast-expanded-seams'
    return candidate
