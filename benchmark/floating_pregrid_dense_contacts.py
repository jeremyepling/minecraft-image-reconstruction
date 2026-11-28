"""Use dense within-side slope controls at appearance borders erased by the baseline."""
import cv2
import numpy as np


def dense_cuts(base, appearance, depth, normal):
    ids=base.copy()
    accepted=[]
    for parent in np.unique(base[base>0]):
        yy,xx=np.nonzero(base==parent)
        box=np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
        original=base[box]==parent
        pieces=appearance[box]
        codes,counts=np.unique(pieces[original],return_counts=True)
        # Retain the largest appearance piece as the original parent.
        for code in codes[np.argsort(-counts)][1:]:
            remaining=ids[box]==parent
            side=remaining & (pieces==code)
            other=remaining & ~side
            if min(side.sum(),other.sum())<80:
                continue
            # Every probe pair must remain in its own photographed region.
            border=side & (cv2.dilate(other.astype('uint8'),np.ones((3,3),'uint8'))>0)
            y,x=np.nonzero(border)
            if len(x)<20:
                continue
            field=cv2.distanceTransform(side.astype('B'),cv2.DIST_L2,5)-cv2.distanceTransform(other.astype('B'),cv2.DIST_L2,5)
            vx=cv2.Sobel(field,cv2.CV_32F,1,0)[y,x]
            vy=cv2.Sobel(field,cv2.CV_32F,0,1)[y,x]
            direction=np.stack((vx,vy),axis=1)/np.maximum(1.e-6,np.hypot(vx,vy))[:,None]
            probe=np.rint(np.stack((x,y),axis=1)[:,None]+direction[:,None]*np.array([-9,-6,-3,3,6,9])[None,:,None]).astype(int)
            valid=((probe>=0)&(probe<original.shape[::-1])).all(axis=(1,2))
            px,py=probe[valid].transpose(2,0,1)
            ok=other[py[:,:3],px[:,:3]].all(axis=1)&side[py[:,3:],px[:,3:]].all(axis=1)
            if ok.sum()<max(16,.45*len(x)):
                continue
            a=depth[box][py[ok],px[ok]]
            near=a[:,3]-a[:,2]
            left=a[:,2]-a[:,0]
            right=a[:,5]-a[:,3]
            jump=near-.5*(left+right)
            med=np.median(jump)
            noise=max(1.e-6,.004*np.median(abs(a)))
            # A continuous slope can have the same sign everywhere: it must also
            # be smaller on the two same-side control spans of equal length.
            n=normal[box][py[ok],px[ok]]
            wins=(np.sign(jump)==np.sign(med)) & (abs(near)>np.maximum(abs(left),abs(right))+noise)
            if abs(med)<noise or wins.mean()<.70 or np.median((n[:,1]*n[:,4]).sum(axis=1))<.93:
                continue
            ids[box][side]=ids.max()+1
            accepted.append(dict(parent=int(parent),appearance=int(code),signed_jump=float(med),probes=int(len(a)),
                                 stronger_than_both=float(wins.mean())))
    return ids,accepted
