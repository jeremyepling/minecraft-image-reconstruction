"""Score the shape matcher on original, frozen Blocker Picker quads."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from engine.atlas import digest
from engine.surface_model import canonical
from engine.blocker_shape import ShapeMatcher


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--atlas',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    matcher=ShapeMatcher(args.atlas)
    artifact=json.loads(args.input.read_bytes())
    rows=[]
    image_path=None
    for row in artifact['rows']:
        if image_path!=row['image']:
            image_path=row['image']
            with Image.open(image_path) as im:
                rgb=np.asarray(im.convert('RGB'))
        result=matcher.rank(rgb,row['quad'])
        frozen = {k:v for k,v in row.items() if k not in ('margin','distances')}
        rows.append(dict(frozen, **result, adjusted_quad=canonical(row['quad']), sampled_quad=canonical(row['quad']), offset=[0,0,1]))
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/'shape.json').write_text(json.dumps(dict(rows=rows,atlas_sha256=artifact['atlas_sha256'],
        inputs_sha256=digest(args.input),matcher_sha256=digest(Path(__file__).parents[1]/'engine/blocker_shape.py'),
        frame_sha256=artifact['frame_sha256']),indent=2))
    print(Counter((r['kind'],r['block_id']==r['truth']) for r in rows))

if __name__=='__main__':
    main()
