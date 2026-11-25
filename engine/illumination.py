"""Assess whether two rectified views of neighboring faces preserve texture under shading."""
import numpy as np

LUMA = np.array([.213, .715, .072], dtype=np.float32)
# Balanced-accuracy threshold selected on the development adjacency wall.
MATERIAL_THRESHOLD = .5049303437647716


def _residual(gray):
    # The local detail survives smooth exposure changes. Ignore the outer rim,
    # where rectification can mix texels from a different material.
    return gray[:,1:-1,1:-1] - .25*(
        gray[:,:-2,1:-1] + gray[:,2:,1:-1] + gray[:,1:-1,:-2] + gray[:,1:-1,2:])


def edge_evidence(first, second, return_gain=False):
    """Compare batches of corresponding rectified face patches in RGB 0..255.

    Material score increases with chromatic and structural change; scalar exposure
    and a smooth or abrupt brightness field can vary across the two inputs.
    """
    a,b = (np.asarray(v,dtype=np.float32)[:,1:-1,1:-1]/255 for v in (first,second))
    if a.shape!=b.shape or a.ndim!=4 or min(a.shape[1:3])<6 or a.shape[-1]!=3:
        raise ValueError('Supply equal batches of RGB patches at least 8 by 8 pixels')
    la,lb = a@LUMA,b@LUMA
    # A small offset regularizes dark texels, where hue and ratios are noisy.
    log_a,log_b = np.log(la+.025),np.log(lb+.025)
    ratio = log_b-log_a
    h,w = ratio.shape[1:]
    x,y = np.meshgrid(np.linspace(-1,1,w),np.linspace(-1,1,h))
    delta = np.median(ratio,axis=(1,2))
    uniform_field = delta[:,None,None]
    flat = abs(ratio-uniform_field).mean((1,2))
    # Fit a planar lighting gradient; penalize extra degrees of freedom.
    centered = ratio-ratio.mean(axis=(1,2),keepdims=True)
    slope_x = (centered*x).sum((1,2))/(x*x).sum()
    slope_y = (centered*y).sum((1,2))/(y*y).sum()
    smooth_field = ratio.mean((1,2),keepdims=True)+slope_x[:,None,None]*x+slope_y[:,None,None]*y
    smooth = abs(ratio-smooth_field).mean((1,2))+.003
    # Abrupt shade can cross a patch at any of three interior fractions.
    step = np.full(len(a),np.inf)
    if return_gain:
        step_field = np.zeros_like(ratio)
    for axis in (1,2):
        r = ratio if axis==1 else ratio.transpose(0,2,1)
        for cut in sorted({round(r.shape[1]*f) for f in (.25,.5,.75)}):
            u,v = r[:,:cut],r[:,cut:]
            before = np.median(u,axis=(1,2),keepdims=True)
            after = np.median(v,axis=(1,2),keepdims=True)
            err = (abs(u-before).sum((1,2))+abs(v-after).sum((1,2)))/(h*w)
            if return_gain:
                improve = err+.005 < step
                dst = step_field if axis==1 else step_field.transpose(0,2,1)
                dst[improve,:cut] = before[improve]
                dst[improve,cut:] = after[improve]
            step = np.minimum(step,err+.005)
    field_cost = np.minimum.reduce([flat,smooth,step])
    model = np.array(['uniform','smooth','step'])[np.argmin([flat,smooth,step],axis=0)]

    # Compare chromaticity independently of the fitted luminance field.
    hue = np.linalg.norm(((a+.025)/(la[...,None]+.025)).mean((1,2))-
                         ((b+.025)/(lb[...,None]+.025)).mean((1,2)),axis=1)
    da,db = _residual(log_a),_residual(log_b)
    source = da[:,1:-1,1:-1]
    correlations = []
    for dy in (0,1,2):
        for dx in (0,1,2):
            target = db[:,dy:dy+source.shape[1],dx:dx+source.shape[2]]
            energy = np.sqrt((source**2).mean((1,2))*(target**2).mean((1,2)))
            correlations.append((source*target).mean((1,2))/(energy+.00005))
    corr = np.maximum.reduce(correlations)
    # Mottled stone with changing hue/texture should not be merged merely
    # because a fitted light field can explain its mean gray level.
    score = .55*(1-np.clip(corr,0,1))+1.4*hue+field_cost
    texture_strength = np.minimum((da**2).mean((1,2))**.5,(db**2).mean((1,2))**.5)
    evidence = dict(material_score=score,luminance_change=abs(delta),log_gain=delta,
                    texture_agreement=corr,chromatic_change=hue,shading_model=model,
                    observable=(texture_strength>.012)|(hue>.05))
    if return_gain:
        # Gain maps the first patch lighting to the second. Extend the fitted
        # interior to the original patch boundary for display. The caller
        # decides whether there is enough texture agreement to apply it.
        log_gain = np.where(model[:,None,None]=='step',step_field,
                     np.where(model[:,None,None]=='smooth',smooth_field,uniform_field))
        evidence['gain_map'] = np.pad(np.exp(np.clip(log_gain,-1.4,1.4)),
                                      ((0,0),(1,1),(1,1)),mode='edge')
    return evidence
