"""Score the shape matcher on original, frozen Blocker Picker quads."""
from engine.face_evidence import MIN_FACE_WIDTH, effective_width
from benchmark.blocker_picker_widths import counts_by_kind
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
        frozen = {k:v for k,v in row.items() if k not in ('margin','distances')}
        width=effective_width(row['quad'])
        if width < MIN_FACE_WIDTH:
            rows.append(dict(frozen, effective_width_px=width, block_id=None, texture=None, ranked_ids=[], adjusted_quad=None, sampled_quad=None, offset=None))
            continue
        result=matcher.rank(rgb,row['quad'])
        rows.append(dict(frozen, **result, effective_width_px=width, adjusted_quad=canonical(row['quad']), sampled_quad=canonical(row['quad']), offset=[0,0,1]))
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/'shape.json').write_text(json.dumps(dict(rows=rows,min_face_width_px=MIN_FACE_WIDTH,counts=counts_by_kind(rows),atlas_sha256=artifact['atlas_sha256'],
        inputs_sha256=digest(args.input),matcher_sha256=digest(Path(__file__).parents[1]/'engine/blocker_shape.py'),
        frame_sha256=artifact['frame_sha256']),indent=2))
    print(Counter((r['kind'], 'under minimum' if r['block_id'] is None else r['block_id']==r['truth']) for r in rows))

if __name__=='__main__':
    main()
