"""Conservative neutral-patch white balance for shader screenshots."""
import numpy as np
from PIL import Image

Y=np.array([.2126,.7152,.0722])

def linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def srgb(rgb):
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1/2.4) - 0.055)


def rect(box, size):
    x,y,r,b = map(int,box)
    if not (0 <= x < r <= size[0] and 0 <= y < b <= size[1]):
        raise ValueError(f"Rectangle {box} outside image {size}")
    return x,y,r,b


def tiles(x0,y0,x1,y1,size):
    """Use multiple phase offsets and include both ends of a narrow feature."""
    w, h = min(size,x1-x0), min(size,y1-y0)
    nx = min(9,1+int(np.ceil((x1-x0-w)/max(w*.7,1))))
    ny = min(9,1+int(np.ceil((y1-y0-h)/max(h*.7,1))))
    for y in np.linspace(y0,y1-h,ny).round().astype(int):
        for x in np.linspace(x0,x1-w,nx).round().astype(int):
            yield x,y,x+w,y+h


def cast_estimate(img, anchors, roi):
    """Neutral-patch white balance in linear RGB. Each anchor gets one vote."""
    candidates = anchors or list(tiles(*roi,max(12,min(roi[2]-roi[0],roi[3]-roi[1])//8)))
    samples = []
    for bounds in candidates:
        a = np.asarray(img.crop(rect(bounds,img.size)).resize((24,24),Image.Resampling.BOX))/255.
        lum = a@Y
        spread = a.max(2)-a.min(2)
        mask = (lum > (0.17 if anchors else .53)) & (a.max(2)<.98) & (spread < .14)
        if mask.mean() < .40:
            continue
        pixels=linear(a[mask])
        log=np.log(np.maximum(pixels,.002))
        v=np.median(log-log[:,1,None],axis=0)
        samples.append((list(map(int,bounds)),v, float(lum[mask].mean())))
    if not samples:
        return np.ones(3), {"method":"identity; insufficient neutral pixels","support":0,"confidence":"low"}
    values=np.array([s[1] for s in samples])
    med=np.median(values,axis=0)
    dev=np.max(np.abs(values-med),axis=1)
    keep=dev<=max(.055,2.5*float(np.median(dev)))
    med=np.median(values[keep],axis=0)
    # Preserve neutral luminance; limit extreme corrections from uncertain anchors.
    light=np.exp(med)
    gains=np.clip((light@Y)/light,.78,1.28)
    dispersion=float(np.median(dev[keep]))
    stats={"method":"linear RGB neutral anchors" if anchors else "automatic bright low-chroma patches",
           "support":int(keep.sum()),"candidates":len(samples),"dispersion_log_rgb":round(dispersion,4),
           "confidence":"moderate" if keep.sum()>=3 and dispersion<.065 else "low",
           "illuminant_rgb_relative_to_green":light.round(4).tolist(),
           "anchors_used":[s[0] for s,ok in zip(samples,keep) if ok]}
    return gains,stats


def correct(img,gains):
    rgb=np.asarray(img,dtype=np.float32)/255.
    return Image.fromarray(np.rint(np.clip(srgb(linear(rgb)*gains),0,1)*255).astype('uint8'))
