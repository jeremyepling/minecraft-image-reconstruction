"""Qualitative no-floor line-camera and local atlas ruler trial on the supplied cottage."""
from hashlib import sha256
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from benchmark.full_block_lattice import line_camera, mask_parts, sprite_candidates
from benchmark.local_wall_rulers import WORK, OUT, choose, read, support
from engine.surface_model import intrinsic


def main():
    folder=WORK/'minecraft-image-reconstruction/runs/cottage-world-20261127'
    source=folder/'source.png'
    image=Image.open(source).convert('RGB')
    rgb=np.asarray(image)
    segments=cv2.createLineSegmentDetector().detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))[0].reshape(-1,4)
    segments=segments[np.linalg.norm(segments[:,2:]-segments[:,:2],axis=1)>14]
    linear=line_camera(segments,image.size)
    stock=read(folder/'main-e3bd220/build-manifest.json')['placement']
    result=dict(source_sha256=sha256(source.read_bytes()).hexdigest(),stock_camera=stock.get('P'),
                line_camera=linear is not None,segments=len(segments),seeds=[],verifications=[])
    pen=ImageDraw.Draw(image)
    parts=read(folder/'main-e3bd220/surfaces.json')['planes']
    if linear is None:
        # Source-only fallback uses measured near-horizontal surface normals and a
        # nominal focal length. A trial can verify photographed tiles locally only.
        normals=[np.array(p['normal_camera']) for p in sorted(parts,key=lambda p:-p.get('support_area',0))
                 if p.get('normal_camera') and abs(p['normal_camera'][1])<.3]
        if normals:
            z=normals[0]
            side=next((n for n in normals if abs(n@z)<.25),None)
            if side is not None:
                x=side-z*(side@z)
                x/=np.linalg.norm(x)
                y=np.cross(z,x)
                y*= -1 if y[1]>0 else 1
                linear=intrinsic(image.size,image.height)@np.column_stack([x,y,z])
                result['fallback']='two source wall normals, centered square pixels, focal = image height'
    if linear is not None:
        lookup={p['id']:p for p in parts}
        for part in [p for p in mask_parts(parts,image.size,linear) if p['axis']!=1][:8]:
            original=lookup[part['id']]
            hull=original.get('sampling_chart')
            if hull:
                outline=[tuple(v) for v in hull]
                pen.line(outline+[outline[0]],fill=(70,150,218),width=1)
            # Bounded discovery windows avoid shrinking an entire broad wall to
            # a tiny chart. All searched pixels are excluded from the later test.
            train=np.zeros_like(part['mask'])
            seeds=[]
            for quantile in (.25,.55,.80):
                x0,y0=np.quantile(part['x'],quantile),np.quantile(part['y'],.32)
                window=part['mask'].copy()
                yy,xx=np.indices(window.shape)
                window &= (abs(xx-x0)<=23)&(abs(yy-y0)<=23)
                if window.sum()<65:
                    continue
                train|=window
                y,x=np.nonzero(window)
                seeds.extend(sprite_candidates(dict(part,mask=window,y=y,x=x,area=len(x)),rgb,linear))
            if not seeds:
                continue
            anchor=max(seeds,key=lambda a:a['corr'])
            q=np.array(anchor['quad'])
            result['seeds'].append(dict(surface=part['id'],quad=q.tolist(),corr=anchor['corr']))
            pen.line([tuple(v) for v in q]+[tuple(q[0])],fill=(255,194,65),width=2)
            # Remove the discovery tile and its two-pixel rim before searching for new rulers.
            mask=support(original,rgb.shape[:2])
            reserved=np.repeat(np.repeat(train,4,0),4,1).astype(np.uint8)
            reserved=cv2.copyMakeBorder(reserved,0,mask.shape[0]-reserved.shape[0],0,mask.shape[1]-reserved.shape[1],cv2.BORDER_CONSTANT)
            mask[cv2.dilate(reserved,np.ones((5,5),np.uint8))>0]=0
            runs=[]
            for y in np.flatnonzero(mask.any(1)):
                limits=np.flatnonzero(np.diff(np.r_[0,mask[y],0]))
                runs.extend((y,a,b) for a,b in zip(limits[::2],limits[1::2]))
            P=np.column_stack([linear,linear@np.array(anchor['xyz'])[0]])
            rec=choose(P,rgb,dict(original,support=dict(runs=runs,size=image.size)),dict(id=part['id'],axis=part['axis'],level=0))
            if rec:
                result['verifications'].append(rec)
                a=rec['arms']['native_mixed']
                q=np.array(a['quad'])
                pen.line([tuple(v) for v in q]+[tuple(q[0])],fill=(40,255,120) if a['accepted'] else (249,90,90),width=3)
    dest=OUT/'cottage'
    dest.mkdir(exist_ok=True,parents=True)
    (dest/'receipt.json').write_text(json.dumps(result))
    image.thumbnail((1050,800))
    image.save(dest/'rulers.jpg',quality=92)
    print('lines',len(segments),'line camera',result['line_camera'],'seeds',len(result['seeds']),'tested',len(result['verifications']))


if __name__=='__main__':
    main()
