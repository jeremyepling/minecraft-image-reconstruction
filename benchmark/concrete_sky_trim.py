"""Check for unhandled daylight sky at native pixels after the released filters."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmark import concrete_shadow_boundaries as trial
from benchmark.pavement_opening_eval import visible_faces
from benchmark.texture_camera_eval import save
from engine.pillar_seams import native_mask
from engine.sky_filter import sky_masks


def sky(image):
    rgb = np.asarray(image).astype(np.int16)
    h,w,_ = rgb.shape
    step = max(1,w//70)
    left = rgb[:,:w//40:step]
    right = rgb[:,-w//40::step]
    template = np.median(np.concatenate((left,right),axis=1),axis=1).astype(np.int16)
    r,g,b = template.T
    sky_rows = (b>=190) & (b-g>=12) & (g-r>=12)
    for side in (left,right):
        sky_rows &= (abs(side-template[:,None]).max(axis=2)<=1).mean(axis=1)>=.9
    # Requiring an unbroken stack of blue margin rows from the top excludes
    # ground colors that appear lower down, including blue staging platforms.
    sky_rows = np.logical_and.accumulate(sky_rows)
    distance = abs(rgb-template[:,None]).max(axis=2)
    similar = (distance<=1) & sky_rows[:,None]
    direct = np.logical_and.accumulate(similar,axis=1) | np.logical_and.accumulate(similar[:,::-1],axis=1)[:,::-1]
    return direct,similar,rgb


def measure(job):
    split,src,_,_ = job
    image,parts,_,_ = trial.viewer.structure_stages(job,trial.baseline()[src['id']])
    strict,blanket,rgb = sky(image)
    owned = np.zeros(strict.shape,bool)
    for part in parts:
        owned |= native_mask(part,image.size)
    labels,rays = visible_faces(split,src)
    clear,cloud = sky_masks(image)
    calibrated = clear | cloud
    row = dict(id=src['id'],split=split,sky_template_pixels=int(blanket.sum()),
               sky_calibration_pixels=int(calibrated.sum()))
    blue = (rgb[:,:,2]-rgb[:,:,0]>20) & (rgb[:,:,2]>=rgb[:,:,1])
    for key,selection in [('blue_color_only',blue),('sky_color_row',blanket),('strict_sky',strict),('calibrated_clear',clear),('calibrated_cloud',cloud)]:
        m = selection & owned
        row[key] = dict(background=int((m & (rays==0)).sum()), visible=int((m & (rays>0)).sum()))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs = list(trial.suite.jobs([*trial.suite.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(measure,jobs))
    save(trial.OUT/'sky-cases.json',rows)
    out={}
    for key,fixed in [('fixed',True),('other',False)]:
        rr=[r for r in rows if (r['split'] in trial.suite.framed.EVALUATION)==fixed]
        out[key]={}
        for mode in ('blue_color_only','sky_color_row','strict_sky','calibrated_clear','calibrated_cloud'):
            counts=Counter()
            for r in rr:
                counts.update(r[mode])
                counts['visible_loss_views']+=int(r[mode]['visible']>0)
                counts['changed_views']+=int(any(r[mode].values()))
            out[key][mode]=dict(counts)
    save(trial.OUT/'sky-summary.json',out)


if __name__=='__main__':
    main()
