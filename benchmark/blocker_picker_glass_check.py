"""Check genuine glass, stone and wood on independently placed registry displays."""
import argparse
import json
from pathlib import Path

from engine.blocker_shape import ShapeMatcher
from engine.geometry import face
from calibration.evaluate import capture
from calibration.projection import camera_project, south_quad


def run(cal, out):
    matcher=ShapeMatcher(cal/'atlas')
    layout=json.loads((cal/'layout.json').read_bytes())
    views=json.loads((cal/'views.json').read_bytes())
    rows=[]
    for view in views:
        if view['split']!='validation' or view.get('kind')=='adjacent':
            continue
        blocks=[b for b in layout['blocks'] if b['page']==view['page']]
        chosen=[b for b in blocks if b['block_id']=='glass' or b['block_id'].endswith('_stained_glass') or b['block_id'] in ('stone_bricks','cracked_stone_bricks','polished_andesite','andesite','tuff','spruce_planks','netherite_block')]
        if not chosen:
            continue
        rgb,pose,verified=capture(cal,view,blocks,'26.3')
        for b in chosen:
            pos=b['x'],b['y'],b['z']
            assert verified[pos]==b['block_id']
            quad=camera_project(south_quad(*pos),pose)
            result=matcher.rank_patch(face(rgb,quad,32))
            rows.append(dict(view=view['id'],truth=b['block_id'],picked=result['block_id']))
    out.write_text(json.dumps(rows,indent=2))
    print(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('calibration',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    run(args.calibration,args.out)
