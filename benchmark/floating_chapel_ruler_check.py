"""Compare the two original sky Chapel ruler cells with visible native face corners."""
import numpy as np
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
    for cell,fid in zip(chart["cells"],ids):
        face=faces[fid]
        up=np.array([0,1,0])
        basis=np.stack([np.cross(up,face["normal_world"]),up])
        uv=np.array(face["corners_world"])@basis.T
        corners=dict(zip(map(tuple,uv-uv.min(0)),face["corners_px"]))
        true=fit_homography([0,0],np.array([corners[tuple(p)] for p in UNIT]))
        proposal=project(np.array(chart["H"]),np.array([cell["u"],cell["v"]])+UNIT)
        mapped=project(np.linalg.inv(true),proposal)
        lengths=np.linalg.norm(mapped-np.roll(mapped,-1,axis=0),axis=1)
        lengths=(lengths[:2]+lengths[2:])/2
        measurements.append(dict(cell=[cell["u"],cell["v"]],predicted_id=cell["block_id"],
            recorded_id=blocks[face["block"]]["block_id"],surface_id=face["surface_id"],face_id=int(fid),
            xyz_visible=face["xyz_world"],size_in_true_blocks=lengths.tolist()))
    dest=ROOT/"benchmark/reviews/floating-patch-controls-01/chapel-ruler.json"
    save(dest,measurements)
    print(measurements)


if __name__=="__main__":
    main()
