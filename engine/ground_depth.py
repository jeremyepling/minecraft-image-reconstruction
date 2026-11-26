"""Measure visible surface planes using image depth calibrated against the visible floor."""
import numpy as np
from PIL import Image

from engine.surface_model import intrinsic
from engine.visible_raster import SAMPLE_STRIDE


def calibrate(field, rays, eye, masks):
    """Fit metric inverse depth on the ground and verify agreement across the image."""
    h, w = field.shape
    y, x = np.indices(field.shape)
    # The side strips can include sky. Use only ground-facing rays within
    # the distance range of the camera solve and below the inferred horizon.
    truth = -rays[:,:,1]/eye[1]
    floor = (~np.logical_or.reduce(masks)) & (y > .53*h) & (truth > .008) & (truth < .6)
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


def plane_levels(image, parts, masks, projection, eye, depth_field):
    if eye[1]<=0 or not masks:
        return {}
    h,w = masks[0].shape
    field = np.asarray(Image.fromarray(np.float32(depth_field)).resize((w,h),Image.Resampling.BILINEAR))
    yy,xx = np.indices(field.shape)
    pixels = np.stack(((xx+.5)*SAMPLE_STRIDE,(yy+.5)*SAMPLE_STRIDE,np.ones_like(xx)),axis=-1)
    rays = pixels @ np.linalg.inv(projection[:,:3]).T
    fit = calibrate(field,rays,eye,masks)
    if fit is None:
        return {}
    inverse = field*fit[0]+fit[1]
    focal = np.linalg.norm(projection[0,:3]-image.width*.5*projection[2,:3])
    rot = np.linalg.solve(intrinsic(image.size,focal),projection[:,:3])
    levels = {}
    for part,mask in zip(parts,masks,strict=True):
        if part.get('normal_camera') is None:
            continue
        axis = int(np.argmax(abs(rot.T @ part['normal_camera'])))
        take = mask & (inverse>.007) & (inverse<.7)
        if take.sum()<8:
            continue
        coords = eye[axis]+rays[:,:,axis][take]/inverse[take]
        level = int(np.rint(np.median(coords)))
        levels[part['id']] = level
    return levels
