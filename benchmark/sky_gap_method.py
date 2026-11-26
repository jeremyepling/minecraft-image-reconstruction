"""Find clear sky through holes using the open sky and skyline in this screenshot."""
import numpy as np

from engine.camera_lines import _runs


METHOD = 'row-sky-with-horizon'
SETTINGS = dict(margin_fraction=.2, color_tolerance=1, min_reference_pixels=4,
                min_area=16, min_rows=4, min_gradient_span=3, horizon_tolerance_fraction=.012,
                horizon_probe_fraction=.008, horizon_ground_tolerance=10, min_horizon_columns=4)


def gap_mask(image, blue, cloud):
    rgb = np.asarray(image.convert('RGB'), dtype=np.int16)
    h,w = blue.shape
    if rgb.shape[:2] != blue.shape or cloud.shape != blue.shape:
        raise ValueError('Expected native-resolution sky masks')
    margin = max(1, round(w*SETTINGS['margin_fraction']))
    sides = np.r_[0:margin, w-margin:w]
    result = np.zeros((h,w),bool)
    # Only rows for which both margins contain previously identified, exposed sky
    # can support a new disconnected region. Never extrapolate their RGB below them.
    near = np.zeros_like(result)
    for y in np.flatnonzero(blue[:,sides[:margin]].sum(1) >= SETTINGS['min_reference_pixels']):
        if blue[y,sides[margin:]].sum() < SETTINGS['min_reference_pixels']:
            continue
        ref = rgb[y,blue[y]]
        tol = SETTINGS['color_tolerance']
        match = np.all((rgb[y]>=ref.min(0)-tol) & (rgb[y]<=ref.max(0)+tol), axis=1)
        near[y] = match & ~blue[y] & ~cloud[y]

    sky_on_sides = blue[:,sides]
    cols = np.flatnonzero(sky_on_sides.any(0))
    if not len(cols):
        return result
    bottom = h-1-np.argmax(sky_on_sides[::-1,cols], axis=0)
    horizon = int(np.median(bottom))
    tolerance = max(2,round(h*SETTINGS['horizon_tolerance_fraction']))
    probe = max(2,round(h*SETTINGS['horizon_probe_fraction']))
    floor_columns = np.abs(bottom-horizon)<=tolerance
    floor = rgb[np.minimum(bottom[floor_columns]+probe,h-1),sides[cols[floor_columns]]]
    ground = np.median(floor, axis=0)
    for runs in _runs(near):
        top,base = runs[0][0],runs[-1][0]
        if base-top+1 < SETTINGS['min_rows'] or sum(z-a for y,a,z in runs)<SETTINGS['min_area']:
            continue
        # To distinguish a solid block from a gradient, the *exposed* sky and
        # the proposed region must both change color in the same vertical direction.
        y0,a0,z0 = runs[0]
        y1,a1,z1 = runs[-1]
        sky_change = np.median(rgb[y1,blue[y1]],axis=0)-np.median(rgb[y0,blue[y0]],axis=0)
        local_change = np.median(rgb[y1,a1:z1],axis=0)-np.median(rgb[y0,a0:z0],axis=0)
        axis = np.argmax(abs(sky_change))
        gradient = abs(sky_change[axis])>=SETTINGS['min_gradient_span'] and abs(local_change[axis]-sky_change[axis])<=2 and abs(local_change[axis])>=SETTINGS['min_gradient_span']
        if not gradient:
            # The noon sky becomes perfectly flat near the horizon. Retain a cut
            # there only if the same sky-to-distant-ground transition is visible.
            if abs(base-horizon)>tolerance or base+probe>=h:
                continue
            columns = np.zeros(w,bool)
            for y,a,z in runs:
                if base-y<=1:
                    columns[a:z] = True
            ground_agrees = np.max(abs(rgb[base+probe,columns]-ground),axis=1)<=SETTINGS['horizon_ground_tolerance']
            if ground_agrees.sum()<SETTINGS['min_horizon_columns'] or ground_agrees.mean()<.5:
                continue
        for y,a,z in runs:
            result[y,a:z] = True
    return result
