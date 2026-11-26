"""Score predicted material transitions against view-masked block and face ownership."""
import argparse
import json
from pathlib import Path

import numpy as np

from engine.atlas_count import load_atlas, prepared_atlas_paths
from engine.geometry import project


def transition(pred, path):
    import cv2
    from PIL import Image

    ev = pred.get("evidence",{})
    if ev.get("transitions"):
        return ev["transitions"][0]
    if not ev.get("transition_pairs"):
        return None
    with Image.open(path/"rgb.png") as opened:
        rgb = np.asarray(opened.convert("RGB"))
    u,v = np.meshgrid((np.arange(16)+.5)/16,1-(np.arange(16)+.5)/16)
    cells = []
    for cell in ev["cells"]:
        world = np.zeros((16,16,3))
        world[...,ev["axis"]] = u+cell["x"]
        world[...,1] = v+cell["y"]
        xy = project(np.array(pred["P"]),world).astype("float32")-.5
        color = cv2.remap(rgb,xy[...,0],xy[...,1],cv2.INTER_LINEAR).mean((0,1))
        cells.append((cell,color))
    pairs = []
    for n,(a,ca) in enumerate(cells):
        for b,cb in cells[n+1:]:
            if (a["texture"]!=b["texture"] and abs(ca-cb).mean()>7
                    and abs(a["x"]-b["x"])+abs(a["y"]-b["y"])==1):
                pairs.append((min(a["score"],b["score"]),[a["x"],a["y"]],[b["x"],b["y"]],
                              a["texture"],b["texture"]))
    return max(pairs) if pairs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    atlas = load_atlas(prepared_atlas_paths("26.3")[0], verify_material_index=True)
    rows = []
    for r in json.loads(args.cases.read_text()):
        pred = r["prediction"]
        ev = pred.get("evidence",{})
        path = args.frames/r["id"]
        pair = transition(pred,path) if pred["P"] else None
        if pair is None:
            continue
        labels = json.loads((path/"labels.json").read_text())
        faces = {f["id"]:f for f in labels["faces"]}
        blocks = {b["id"]:b["block_id"] for b in labels["blocks"]}
        with np.load(path/"labels.npz") as z:
            indices = z["face_id"]
        # Count once per image, using the prediction's highest-scoring pair.
        score,a,b,t1,t2 = pair
        world = np.zeros((2,3))
        world[:,ev["axis"]] = np.array([a[0],b[0]])+.5
        world[:,1] = np.array([a[1],b[1]])+.5
        xy = np.floor(project(np.array(pred["P"]),world)).astype(int)
        observed = [faces.get(int(indices[y,x])) for x,y in xy]
        actual = [blocks[f["block"]] if f else None for f in observed]
        ok = []
        for texture,face,block in zip((t1,t2),observed,actual):
            candidates = atlas.ids[atlas.labels.index(texture)]
            ok.append(block in candidates)
        adjacent = (all(observed) and observed[0]["direction"]==observed[1]["direction"] and
                    np.abs(np.subtract(observed[0]["xyz_world"],observed[1]["xyz_world"])).sum()==1)
        rows.append(dict(id=r["id"],predicted_texture=[t1,t2],actual_block=actual,
                         score=score,exact_block_compatible=ok,adjacent_coplanar=bool(adjacent)))
    result = dict(frames=len(rows),both_compatible=sum(all(r["exact_block_compatible"]) for r in rows),
                  distinct_visible_blocks=sum(None not in r["actual_block"] and len(set(r["actual_block"]))==2 for r in rows),
                  adjacent=sum(r["adjacent_coplanar"] for r in rows), rows=rows)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k!="rows"})


if __name__ == "__main__":
    main()

