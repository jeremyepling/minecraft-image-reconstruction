"""Compare voxel occupancy and in-game previews for the same source, texel and remapped builds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.color_gallery import load
from benchmark.live_color_gallery import frame, label_font, thumbnail


def aligned(rows):
    coords=np.array([[b[axis] for axis in ('x','y','z')] for b in rows],dtype=int)
    lo,hi=coords.min(0),coords.max(0)
    # The front faces south. Anchor the front surface as well as the left
    # edge and ground line, without optimizing registration from source labels.
    coords-=np.array([lo[0],lo[1],hi[2]])
    return {tuple(map(int,p)) for p in coords}, (hi-lo+1).tolist()


def spatial_score(source, reconstruction):
    ref,ref_size=aligned(source)
    observation,size=aligned(reconstruction)
    scores={}
    for name,axes in [('front_xy',(0,1)),('footprint_xz',(0,2)),('occupied_xyz',(0,1,2))]:
        src={tuple(p[i] for i in axes) for p in ref}
        pred={tuple(p[i] for i in axes) for p in observation}
        intersection=len(src & pred);union=len(src | pred)
        scores[name]=dict(source=len(src),predicted=len(pred),intersection=intersection,
                          union=union,iou=intersection/union)
    return dict(size_xyz=size,source_size_xyz=ref_size,occupancy=scores)


def preview(directory,view,source_blocks):
    folder=directory/view
    expected=load(folder/'build/placements.json')['blocks']
    def to_map(data):
        bottom=min(b['y'] for b in data)
        return {(b['x'],b['y']-bottom,b['z']):b['block_id'] for b in data}
    exported=to_map(expected)
    # A revised pack can have its own fresh verification alongside older runs.
    # Match to the current export before choosing a screenshot.
    for run in [folder/'verify',*sorted(folder.glob('verify-*'))]:
        if not (run/'verification.json').is_file(): continue
        verification=load(run/'verification.json')
        if verification['differences'] or verification['matched']!=len(expected): continue
        actual=load(run/'captures/reconstruction-blocks.json')
        if len(actual)==len(expected) and to_map(actual)==exported: break
    else:
        raise ValueError(f'No verified framebuffer matches the current export in {folder}')
    image,mask,_=frame(run/'captures','reconstruction',actual)
    score=spatial_score(source_blocks,actual)
    score['readback']={'expected':len(expected),'matched':len(actual),'run':run.name}
    return thumbnail(image,mask),score


def compare(args):
    views={v['id']:v for v in load(args.fixture/'views.json')}
    blocks=defaultdict(list)
    for b in load(args.fixture/'layout.json')['blocks']:
        blocks[b['page']].append(b)
    ids=args.views or [v for v in views if (args.texel/v/'verify').exists() and (args.remap/v/'verify').exists()]
    if not ids:raise ValueError('No common verified views')
    canvas=Image.new('RGB',(1152,52+260*len(ids)),'#1f2428')
    draw=ImageDraw.Draw(canvas); header=label_font(20); label=label_font(15)
    scores={}
    for i,title in enumerate(('Source','Previous geometry','Revised geometry')):
        draw.text((i*384+6,10),title,fill='white',font=header)
    for row,view in enumerate(ids):
        case=views[view];truth=blocks[case['page']]
        if case['angle']!='front':
            raise ValueError(f'{view} is not a front view; an axis-aligned front comparison requires one')
        im,mask,_=frame(args.fixture/'captures',view,truth)
        source_size=aligned(truth)[1];summaries={}
        panels=[(thumbnail(im,mask),source_size,None)]
        for tag,path in [('texel',args.texel),('remap',args.remap)]:
            photo,result=preview(path,view,truth)
            panels.append((photo,result['size_xyz'],result['occupancy']['front_xy']['iou']))
            summaries[tag]=result
        scores[view]=summaries
        for i,(pic,size,overlap) in enumerate(panels):
            x,y=i*384,48+row*260
            canvas.paste(pic,(x+(384-pic.width)//2,y+(193-pic.height)//2))
            w,h,d=size
            draw.text((x+6,y+198),f'{w} wide × {h} high × {d} deep',font=label,fill='white')
            note=case['kind'].replace('_',' ').capitalize() if overlap is None else f'Front occupancy IoU {overlap:.2f}'
            draw.text((x+6,y+220),note,font=label,fill='#c7cfd3')
    report=dict(alignment='Translate min X and min Y to zero and max Z (south front) to zero; no rotation, scaling or search.',
                previews='Live framebuffer crops are framed separately for legibility. Dimensions are in world blocks.',
                cases=scores)
    for path in (args.image,args.metrics):path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.image,quality=92,subsampling=0)
    args.metrics.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','texel','remap','image','metrics'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--views',nargs='+')
    compare(p.parse_args())
