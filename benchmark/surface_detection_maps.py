"""Evaluation-only native surface maps and one-to-one detection records for review images."""
import numpy as np

from benchmark.camera_labels import rasterize, screen_matrix
from benchmark.camera_round import read
from benchmark.count_metrics import reference_planes
from benchmark.mask_metrics import PROTOCOL, footprint, measure
from benchmark.surface_metrics import _one_to_one
from benchmark.surface_reference import recorded
from engine.surface_perimeter import _native_alpha


def _native_reference(pose,reference,faces):
    """Replay the original label projection, preserving face order for ties at pixel centers."""
    width,height=pose['image_size']
    eye=np.asarray(pose['eye_world'])
    projection=screen_matrix(pose)
    planes=reference_planes(reference,faces)
    owners={(tuple(f['xyz']),f['direction']):pid for pid,plane in planes.items() for f in plane['faces'].values()}
    face_ids=np.zeros((height,width),np.int32)
    depths=np.zeros((height,width),float)
    by_face=[0]
    for index,face in enumerate(faces,1):
        by_face.append(owners[tuple(face['xyz']),face['direction']])
        corners=np.asarray(face['world_corners'])
        result=rasterize((corners-eye)@projection[:,:3].T+projection[:,3],width,height)
        if result is None:
            raise ValueError('A face labeled visible is outside the native frame')
        box,inside,depth,_,_=result
        front=inside&(depth>depths[box])
        face_ids[box][front]=index
        depths[box][front]=depth[front]
    counts=np.bincount(face_ids.ravel(),minlength=len(by_face))[1:]
    if not np.array_equal(counts,[f['visible_pixels'] for f in faces]):
        raise ValueError('Native replay did not reproduce the pixel count of every labeled block face')
    return np.asarray(by_face,dtype=np.int32)[face_ids]


def _intersections(parts,reference,native):
    """Raw prediction intersections include background (column zero)."""
    stride_ids=reference['id_map']
    size=reference['image_size']
    count=max(reference['patches'],default=0)+1
    sampled=np.zeros((len(parts),count),np.int64)
    full=np.zeros_like(sampled)
    predicted=np.zeros_like(native)
    for i,part in enumerate(parts):
        mask=footprint(part,stride_ids.shape,size,True)
        sampled[i]=np.bincount(stride_ids[mask],minlength=count)
    # Match the scorer's ownership precedence for overlaps.
    for i in sorted(range(len(parts)),key=lambda i:-int(sampled[i].sum())):
        crop=_native_alpha(parts[i]['support'],size)
        if crop is None:
            continue
        alpha,(x,y,xx,yy)=crop
        inside=np.asarray(alpha)>0
        full[i]=np.bincount(native[y:yy,x:xx][inside],minlength=count)
        predicted[y:yy,x:xx][inside]=i+1
    return predicted,sampled,full


def _ratios(hits,target_sizes):
    unions=hits.sum(axis=1,keepdims=True)+target_sizes[None,:]-hits
    return np.divide(hits,unions,out=np.zeros(hits.shape,float),where=unions>0)


def detection_maps(case,parts,scores):
    """Return native reference IDs, native proposal indices (one-based), and JSON review data.

    Regular matches are the saved benchmark protocol on its four-pixel grid. Small
    references get a secondary native-pixel 50% IoU match among remaining proposals.
    """
    pose,ref=recorded(case)
    verified,pairs=measure(parts,ref,use_mask=True)
    if verified.keys()!=scores.keys() or not all(np.isclose(v,scores[k],rtol=1e-12,atol=1e-9) for k,v in verified.items()):
        raise ValueError('Surface scores differ from the saved run')
    faces=read(case.path('labels'))['faces']
    native=_native_reference(pose,ref,faces)
    proposed,sampled,full=_intersections(parts,ref,native)
    patches=ref['patches']
    n=full.shape[1]
    native_sizes=np.bincount(native.ravel(),minlength=n)
    sample_sizes=np.bincount(ref['id_map'].ravel(),minlength=n)
    native_iou=_ratios(full,native_sizes)
    sampled_iou=_ratios(sampled,sample_sizes)
    official=dict(pairs)
    small=[pid for pid,patch in patches.items() if not patch['eligible']]
    small_pairs=[(native_iou[i,pid],i,pid) for i in range(len(parts)) if i not in official
                 for pid in small if native_iou[i,pid]>=PROTOCOL['iou']]
    extra_matches=dict(_one_to_one(small_pairs))
    all_pairs={**official,**extra_matches}
    matched_by={pid:i for i,pid in all_pairs.items()}
    references=[]
    for pid,patch in patches.items():
        regular=patch['eligible']
        hits=sampled if regular else full
        ious=sampled_iou if regular else native_iou
        sizes=sample_sizes if regular else native_sizes
        i=matched_by.get(pid)
        best=int(hits[:,pid].argmax()) if hits.shape[0] and hits[:,pid].any() else None
        if regular:
            status='found' if i is not None else 'missed'
        else:
            status='small_found' if i is not None else 'small'
        overlap=int(hits[best,pid]) if best is not None else 0
        coverage=overlap/int(sizes[pid])
        reason=None
        if status=='missed':
            if best is None:
                reason='no predicted pixels on this surface'
            elif best in official:
                reason='overlapping proposal matched a different surface'
            else:
                reason='overlap below the one-to-one match threshold'
        references.append(dict(id=pid,status=status,eligible=regular,faces=patch['faces'],
            pixels=int(native_sizes[pid]),samples=int(sample_sizes[pid]),
            prediction=i+1 if i is not None else None,
            match_iou=float(ious[i,pid]) if i is not None else None,
            best_prediction=best+1 if best is not None else None,best_prediction_matched_ref=official.get(best),
            coverage=coverage,iou=float(ious[best,pid]) if best is not None else 0.,
            overlap_pixels_or_samples=overlap,overlap_grid='stride-4' if regular else 'native',reason=reason))
    predictions=[]
    for i,part in enumerate(parts):
        pid=all_pairs.get(i)
        total=int(sampled[i].sum())
        background=float(sampled[i,0])/total if total else None
        status='found' if i in official else 'small_found' if i in extra_matches else 'extra'
        reason=None
        if status=='extra':
            reason='mostly background' if not total or background>.5 else 'fragment or mixed surface'
        predictions.append(dict(id=i+1,proposal=part['id'],status=status,reference=pid,
                                background_fraction=background,pixels=int(full[i].sum()),reason=reason))
    counts=dict(visible=len(patches),scored=verified['eligible'],found=len(pairs),missed=verified['eligible']-len(pairs),
                small=len(small),small_found=len(extra_matches),unsampled=sum(int(sample_sizes[p]==0) for p in small),
                proposed=len(parts),unmatched=len(parts)-len(pairs),unpaired=len(parts)-len(all_pairs))
    return native,proposed,dict(counts=counts,references=references,predictions=predictions)
