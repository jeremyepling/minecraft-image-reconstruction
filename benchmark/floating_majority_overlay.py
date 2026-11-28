"""Draw the annotated upper row and measure its top corners against frontmost brick faces."""
import numpy as np
from PIL import Image, ImageDraw
import cv2

from benchmark.floating_majority_edge import (
    PRIOR, REVIEW, RUN, SCALE, TESTS, UNIT, faces_at_depth, paths, pixels, project, read, save,
)


def main():
    panel,name,wall=TESTS[0]
    case=next(c for c in paths(panel)[0][-1] if c["id"]==name)
    image=np.array(Image.open(case["variants"]["sky"]["image"]).convert("RGB"))
    from pathlib import Path
    frame=Path(case["frame"])
    with np.load(frame/"labels.npz") as d:
        ids=d["face_id"]
    labels=read(frame/"labels.json")
    faces={f["id"]:f for f in labels["faces"]}
    material={b["id"]:b["block_id"] for b in labels["blocks"]}
    planes=faces_at_depth(labels)
    source=read(REVIEW/(name+".json.gz"))
    keys=[c["uv"] for c in source["truth"]["seam"]["rows"]]
    visible_brick=np.isin(ids,[fid for fid,f in faces.items() if f["direction"]=="east" and material[f["block"]]=="minecraft:bricks"])
    sandstone=np.isin(ids,[fid for fid,f in faces.items() if "sandstone" in material[f["block"]]])
    junction=visible_brick & (cv2.dilate(sandstone.astype("uint8"),np.ones((3,3),"uint8"))>0)
    seam_distance=cv2.distanceTransform((~junction).astype("uint8"),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    images=[]
    measured={}
    top=np.array([[0,1],[1,1]])
    for arm,root in (("prior",PRIOR/panel/"row-with-material"),("seam",RUN/panel/"seam")):
        chart=next(c for c in read(root/"charts/sky"/name/"charts.json")["charts"] if c["id"]==wall)
        h=np.asarray(chart["H"])
        materials={(p["u"],p["v"]):p["block_id"] for p in chart["cells"]}
        render=Image.fromarray(image)
        draw=ImageDraw.Draw(render)
        points=[]
        measurements=[]
        for key in keys:
            wire=project(h,UNIT+key)
            draw.line([tuple(p) for p in np.vstack((wire,wire[0]))],fill="#00faff",width=1)
            points.extend(wire)
            x,y,n=pixels(h,key,ids.shape)
            candidates=[(cnt,fid) for fid,cnt in zip(*np.unique(ids[y,x],return_counts=True)) if
                        fid in faces and material[faces[fid]["block"]]=="minecraft:bricks" and
                        faces[fid]["direction"]=="east"]
            _,fid=max(candidates)

            face=faces[fid]
            projected=project(h,top+key)
            corners=np.asarray(face["corners_px"])
            upper=np.asarray(face["corners_world"])[:,1]==max(p[1] for p in face["corners_world"])
            # All top corners are projected; record native visibility separately.
            distances=np.linalg.norm(projected[:,None]-corners[None,upper],axis=2)
            top_line=np.floor(project(h,np.column_stack((np.linspace(.05,.95,21),np.ones(21)))+key)).astype(int)
            measurements.append(dict(visible_boundary_distance_px=seam_distance[top_line[:,1],top_line[:,0]].tolist(),uv=key,emitted=materials[tuple(key)],dominant_face=int(fid),
                plane=planes[fid],top_corner_visible=np.array(face["corner_visible"])[upper].tolist(),
                nearest_top_corner_px=distances.min(axis=1).tolist()))
        points=np.array(points)
        lo=np.floor(points.min(axis=0)-15).astype(int)
        hi=np.ceil(points.max(axis=0)+15).astype(int)
        images.append((arm,render,lo,hi))
        measured[arm]=measurements
    low=np.min([im[2] for im in images],axis=0)
    high=np.max([im[3] for im in images],axis=0)
    box=tuple(np.r_[low,high])
    edge=np.zeros(ids.shape,bool)
    edge[1:]|=ids[1:]!=ids[:-1]
    edge[:,1:]|=ids[:,1:]!=ids[:,:-1]
    gold=edge & (ids!=0)
    panels=[("Source",Image.fromarray(image))]
    for arm,im,_,_ in images:
        arr=np.array(im)
        cyan=(arr==[0,250,255]).all(axis=2)
        arr[gold]=[255,199,40]
        arr[gold & cyan]=255
        panels.extend([(arm+" grid",im),(arm+" + visible reference",Image.fromarray(arr))])
    w,h=(high-low)*2
    sheet=Image.new("RGB",(int(w),int((h+24)*len(panels))),"#162031")
    d=ImageDraw.Draw(sheet)
    for i,(title,im) in enumerate(panels):
        d.text((8,int(i*(h+24)+4)),title,fill="white")
        sheet.paste(im.crop(box).resize((int(w),int(h)),Image.Resampling.NEAREST),(0,int(i*(h+24)+24)))
    sheet.save(REVIEW/"twinspire-row.png")
    save(REVIEW/"twinspire-upper-corners.json",measured)


if __name__=="__main__":
    main()

