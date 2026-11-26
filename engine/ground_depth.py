"""Measure visible surface planes using image depth calibrated against the visible floor."""
import numpy as np
from PIL import Image

from engine.surface_model import intrinsic
from engine.visible_raster import SAMPLE_STRIDE


def calibrate(field, rays, eye, masks, *, inverse=True):
    """Fit metric depth on the ground and verify agreement across the image."""
    h, w = field.shape
    y, x = np.indices(field.shape)
    # The side strips can include sky. Use only ground-facing rays within
    # the distance range of the camera solve and below the inferred horizon.
    inverse_ground = -rays[:,:,1]/eye[1]
    floor = (~np.logical_or.reduce(masks)) & (y > .53*h) & (inverse_ground > .008) & (inverse_ground < .6)
    truth = inverse_ground if inverse else 1./np.maximum(inverse_ground,.001)
    fits = []
    for side in (x < w*.18, x > w*.82):
        obs = []
        for band in np.array_split(np.arange(int(.53*h)+1,h), 18):
            take = (floor & side)[band]
            if take.sum() < 12:
                continue
            obs.append((np.median(field[band][take]), np.median(truth[band][take])))
        if len(obs)<8:
            return None
        values, target = np.array(obs).T
        if np.ptp(values)<1e-6 or np.ptp(target)<.015:
            return None
        design = np.column_stack((values,np.ones_like(values)))
        fit = np.linalg.lstsq(design,target,rcond=None)[0]
        if fit[0]<=0 or np.quantile(abs(design@fit-target),.8)>.12*np.ptp(target):
            return None
        fits.append(fit)
    if abs(fits[0][0]-fits[1][0])>.18*max(fits[0][0],fits[1][0]):
        return None
    return np.mean(fits,axis=0)


def _contact_anchors(parts, masks, rays, eye, rotation, estimated_depth):
    h,w=masks[0].shape
    y=np.arange(h)[:,None]
    union=np.logical_or.reduce(masks)
    bottom=union & (y==np.where(union,y,-1).max(0)) & (y<h-1)
    foot_z=np.divide(-eye[1],rays[:,:,1],out=np.full((h,w),np.nan),where=rays[:,:,1]<-1e-5)
    anchors,scales={},[]
    for part,mask in zip(parts,masks,strict=True):
        if part.get('normal_camera') is None or not mask.any(): continue
        axis=int(np.argmax(abs(rotation.T@part['normal_camera'])))
        foot=bottom & mask & (foot_z>1) & (foot_z<120)
        if axis==1 or foot.sum()<max(7,.24*mask.any(0).sum()):continue
        samples=eye[axis]+foot_z[foot]*rays[:,:,axis][foot]
        level=round(np.median(samples))
        if np.quantile(abs(samples-level),.80)>.38:continue
        take=mask & (abs(rays[:,:,axis])>.1) & (estimated_depth>1.5)
        if take.sum()<12:continue
        truth=(level-eye[axis])/rays[:,:,axis][take]
        ratios=truth/estimated_depth[take]
        scale=np.median(ratios)
        if not .82<scale<1.18 or np.quantile(abs(ratios-scale),.75)>.035:continue
        anchors[part['id']]=int(level)
        scales.append(scale)
    if scales and np.ptp(scales)<.115:
        return anchors,np.median(scales)
    return {},1.


def plane_measurements(image, parts, masks, projection, eye, depth_field, *, inverse=True, anchored=False):
    if eye[1]<=0 or not masks:
        return {}
    h,w = masks[0].shape
    field = np.asarray(Image.fromarray(np.float32(depth_field)).resize((w,h),Image.Resampling.BILINEAR))
    yy,xx = np.indices(field.shape)
    pixels = np.stack(((xx+.5)*SAMPLE_STRIDE,(yy+.5)*SAMPLE_STRIDE,np.ones_like(xx)),axis=-1)
    rays = pixels @ np.linalg.inv(projection[:,:3]).T
    fit = calibrate(field,rays,eye,masks,inverse=inverse)
    if fit is None:
        return {}
    fitted = field*fit[0]+fit[1]
    inverse_depth = fitted if inverse else 1./np.maximum(fitted,.001)
    focal = np.linalg.norm(projection[0,:3]-image.width*.5*projection[2,:3])
    rot = np.linalg.solve(intrinsic(image.size,focal),projection[:,:3])
    anchors, scale = _contact_anchors(parts,masks,rays,eye,rot,1./np.maximum(inverse_depth,.001)) if anchored else ({},1.)
    levels = {}
    for part,mask in zip(parts,masks,strict=True):
        if part.get('normal_camera') is None:
            continue
        axis = int(np.argmax(abs(rot.T @ part['normal_camera'])))
        take = mask & (inverse_depth>.007) & (inverse_depth<.7)
        if take.sum()<8:
            continue
        coords = eye[axis]+scale*rays[:,:,axis][take]/inverse_depth[take]
        middle = anchors.get(part['id'],float(np.median(coords)))
        levels[part['id']] = (middle,float(np.median(abs(coords-middle))))
    return levels


def plane_levels(*args, **kwargs):
    return {key:round(center) for key,(center,_) in plane_measurements(*args,**kwargs).items()}
