"""Compare the two original sky Chapel ruler cells with visible native face corners."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from benchmark.floating_patch_controls import ROOT, Path, read, save, inputs, native_centers, fit_homography, project, UNIT


def main():
    _,ordinary,_,cases=inputs("sky")
    case=next(c for c in cases if "chapel" in c["id"])
    frame=Path(case["frame"])
    labels=read(frame/"labels.json")
    faces={f["id"]:f for f in labels["faces"]}
    blocks={b["id"]:b for b in labels["blocks"]}
    chart=next(c for c in read(ordinary/"sky"/case["id"]/"charts.json")["charts"] if c["id"]=="mask-0")
    with np.load(frame/"labels.npz") as arrays:
        _,ids=native_centers(chart,arrays["face_id"])
    measurements=[]
    outlines=[]
    for cell,fid in zip(chart["cells"],ids):
        face=faces[fid]
        up=np.array([0,1,0])
        basis=np.stack([np.cross(up,face["normal_world"]),up])
        uv=np.array(face["corners_world"])@basis.T
        corners=dict(zip(map(tuple,uv-uv.min(0)),face["corners_px"]))
        true=fit_homography([0,0],np.array([corners[tuple(p)] for p in UNIT]))
        proposal=project(np.array(chart["H"]),np.array([cell["u"],cell["v"]])+UNIT)
        outlines.append((proposal,np.asarray(face["corners_px"])))
        mapped=project(np.linalg.inv(true),proposal)
        lengths=np.linalg.norm(mapped-np.roll(mapped,-1,axis=0),axis=1)
        lengths=(lengths[:2]+lengths[2:])/2
        measurements.append(dict(cell=[cell["u"],cell["v"]],predicted_id=cell["block_id"],
            recorded_id=blocks[face["block"]]["block_id"],surface_id=face["surface_id"],face_id=int(fid),
            xyz_visible=face["xyz_world"],size_in_true_blocks=lengths.tolist()))
    dest=ROOT/"benchmark/reviews/floating-patch-controls-01/chapel-ruler.json"
    save(dest,measurements)
    source=Image.open(case["variants"]["sky"]["image"]).convert("RGB")
    points=np.vstack([p for pair in outlines for p in pair])
    cx,cy=(points.min(0)+points.max(0))*.5
    span=max(np.ptp(points,axis=0))*1.8
    box=tuple(map(int,(cx-span/2,cy-span/2,cx+span/2,cy+span/2)))
    zoom=source.crop(box).resize((320,320),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(zoom)
    for color,index in (("#28baf9",1),("#ff9235",0)):
        for pair in outlines:
            poly=(pair[index]-np.array(box[:2]))*320/(np.array(box[2:])-box[:2])
            pts=[tuple(p) for p in poly]
            draw.line(pts+pts[:1],fill=color,width=3)
    ImageDraw.Draw(source).rectangle(box,outline="white",width=3)
    source.thumbnail((500,320),Image.Resampling.LANCZOS)
    sheet=Image.new("RGB",(840,398),"#172332")
    font=ImageFont.truetype("arial.ttf",17)
    draw=ImageDraw.Draw(sheet)
    draw.text((8,8),"Chapel · floating-sky source image",font=font,fill="white")
    draw.text((512,8),"Seed measurements at the white box",font=font,fill="white")
    sheet.paste(source,(4,36))
    sheet.paste(zoom,(512,36))
    draw.text((8,365),"Orange: proposed one-block cells",font=font,fill="#ff9235")
    draw.text((390,365),"Blue: recorded one-block calcite faces",font=font,fill="#28baf9")
    sheet.save(dest.with_name("chapel-ruler.jpg"),quality=93)
    print(measurements)


if __name__=="__main__":
    main()
