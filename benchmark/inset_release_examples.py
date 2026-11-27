"""Verify ordinary exports and mark observed inset backs on other complete structures."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import facade_inset_round as suite
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from engine.pillar_seams import native_mask


NAMES={'corner-mixed-26.3--heldout-facade-oblique-left':('corner-learned','Corner Facade'),
       'train-sawtooth3-cool-high-right':('saw-learned','Cool Sawtooth'),
       'entrance-confirm-26.3--positive--heldout-facade-oblique-right':('learned','Facade, second view'),
       'heldout-facade-oblique-right':('focus','Facade'),
       'train-sawtooth3-earth-high-right':('saw','Sawtooth 3'),
       'test-recess-28-left':('recess','Recess 28'),
       'signed-relief-26.3--dev--dev-relief-00-oblique-right':('relief','Dev Relief 00')}


def mark(photo,mask,color):
    rgb=np.array(photo)
    padded=np.pad(mask,1)
    center=padded[1:-1,:-2]&padded[1:-1,2:]&padded[:-2,1:-1]&padded[2:,1:-1]
    rgb[mask]=.47*rgb[mask]+.53*np.array(color)
    rgb[mask&~center]=color
    return Image.fromarray(rgb)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/facade-inset-round-01'))
    args=parser.parse_args()
    board=Image.new('RGB',(1236,405),'#162732')
    draw=ImageDraw.Draw(board)
    draw.text((14,9),'Other photographs with recovered recessed walls',font=font(24),fill='white')
    draw.text((15,44),'Green marks backs measured from a side and sill. Violet marks the additional texture match.',
              font=font(16),fill='#dbebef')
    entries=[]
    local=Path('benchmark/reviews/facade-local-placement-01')
    with ZipFile(checked(local/'exports.zip',read(local/'summary.json')['exports_sha256'])) as z:
        for job in suite.jobs(list(suite.framed.SIZES)):
            split,f,_,control=job
            key=f['id']
            if key not in NAMES:
                continue
            short,name=NAMES[key]
            root=args.runs/('production-'+short)
            im,_,_,_=suite.ordinary_inputs(*job)
            manifest=read(root/'build-manifest.json')
            assert manifest['through']=='placement' and manifest['image_sha256']==f['files']['rgb.png']
            actual=decoded_pack(root/'reconstruction.zip')
            member='clear_front/'+split+'/'+key+'/reconstruction.zip'
            target=(decoded_pack(BytesIO(z.read(member))) if member in z.namelist() else
                    decoded_pack(checked(suite.CONTROL/split/key/'reconstruction.zip',control['pack_sha256'])))
            assert actual==target
            surfaces=read(root/'surfaces.json')['planes']
            byid={p['id']:native_mask(p,im.size) for p in surfaces}
            section=manifest['surface_finder']['boundary_settings']['visible_insets']
            first=section['events']
            extra=section.get('texture_events',[])
            outlines=section['count_confirmed_outlines']
            counts=manifest['placement'].get('inset_placement',dict(added=0,removed=0))
            row=dict(frame=key,rgb_sha256=f['files']['rgb.png'],block_count=len(actual),
                     golden_pack_equal=True,return_backs=len(first),texture_backs=len(extra),count_confirmed_outlines=len(outlines),
                     measured_counts={r['id']:r['observed_visible_grid_span'] for r in read(root/'counts.json')['surfaces'] if 'observed_visible_grid_span' in r},
                     added=counts['added'],removed=counts['removed'],stages={n:digest(root/n)
                     for n in ('surfaces.json','counts.json','placements.json','reconstruction.zip')})
            if short=='focus':
                standalone=args.runs/'production-surface-only'
                assert digest(standalone/'surfaces.json')==row['stages']['surfaces.json']
                a,b=root/'surface-crops',standalone/'surface-crops'
                crops=sorted(p.name for p in a.glob('*.png'))
                assert crops==sorted(p.name for p in b.glob('*.png'))
                assert all(digest(a/c)==digest(b/c) for c in crops)
                row.update(standalone_surfaces_identical=True,standalone_crops_identical=len(crops))
            if short=='learned':
                standalone=args.runs/'production-learned-counts'
                assert len(outlines)==2 and all(e['depth']=='unresolved' for e in outlines)
                for stage in ('surfaces.json','counts.json'):
                    assert digest(standalone/stage)==row['stages'][stage]
                row['standalone_counts_and_surfaces_identical']=True
                annotated=im
                for event in outlines:
                    annotated=mark(annotated,byid[event['output']],(255,190,62))
                annotated.save(args.out/'learned-outlines.jpg',quality=95)
            entries.append(row)
            if short not in ('saw','recess','relief'):
                continue
            marked=im
            for event in first:
                marked=mark(marked,byid[event['output']],(72,235,170))
            for event in extra:
                marked=mark(marked,byid[event['output']],(216,132,255))
            all_masks=np.logical_or.reduce(list(byid.values()))
            y,x=np.nonzero(all_masks)
            box=(max(0,int(x.min())-15),max(0,int(y.min())-15),min(im.width,int(x.max())+16),min(im.height,int(y.max())+16))
            left=12+410*('saw','recess','relief').index(short)
            draw.text((left,82),name,font=font(20),fill='white')
            thumb=ImageOps.contain(marked.crop(box),(397,220),Image.Resampling.LANCZOS)
            board.paste(thumb,(left+(397-thumb.width)//2,117+(220-thumb.height)//2))
            draw.text((left,352),f"{counts['added']} correct blocks recovered",font=font(16),fill='#a6f1cc')
            draw.text((left,377),f"{counts['removed']} false foreground blocks removed",font=font(15),fill='#e0e8eb')
    assert len(entries)==len(NAMES)
    args.out.mkdir(parents=True,exist_ok=True)
    board.save(args.out/'other-buildings.jpg',quality=94)
    save(args.out/'ordinary-entry.json',entries)


if __name__=='__main__':
    main()
