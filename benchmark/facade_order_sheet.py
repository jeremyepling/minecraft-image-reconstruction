"""Show exact photographed pixels of newly recovered visible cells on fixed benchmark cases."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, read
from benchmark.texture_camera_sheet import font


EXAMPLES = {
    'shape-benchmark-26.3--train-canopy-06-left': 'Canopy 06',
    'inverse-voxels-26.3--confirm--test-bays-02-oblique-right': 'Bays 02',
    'signed-relief-26.3--test--test-relief-00-oblique-left': 'Relief 00',
}


def draw(run, baseline, out):
    def rows(folder):
        return {r['id']: r for r in read(checked(folder/'cases.json', read(folder/'summary.json')['cases_sha256']))}
    scores, previous = rows(run), rows(baseline)
    page = Image.new('RGB', (1290, 704), '#102128')
    pen = ImageDraw.Draw(page)
    pen.text((20, 12), 'Where the final facade check adds a block', font=font(31), fill='white')
    pen.text((21, 58), 'Top: full photograph. Bottom: structure crop. Cyan marks source pixels of newly correct blocks.', font=font(19), fill='#ddf1f1')
    def put(image, x, y, height):
        scaled = ImageOps.contain(image, (406,height), Image.Resampling.LANCZOS)
        page.paste(scaled, (x+(406-scaled.width)//2, y))
    for split, f, old in framed.cases(('old',)):
        key=f['id']
        if key not in EXAMPLES:
            continue
        _, _, _, _, _, _, _, _, _, before, _ = read_view(baseline, split, f, old, previous[key])
        image, _, _, _, _, labels, ids, truth, _, after, _ = read_view(run, split, f, old, scores[key])
        gained = (truth & after) - before
        assert len(gained) == scores[key]['relative_geometry']['gained']
        shown = np.array(image)
        marked=np.isin(ids, [face['id'] for face in labels['faces'] if tuple(face['xyz_world']) in gained])
        shown[marked]=(.35*shown[marked]+.65*np.array([0,255,226])).astype('uint8')
        overlay=Image.fromarray(shown)
        bbox=Image.fromarray(ids>0).getbbox()
        x0,y0,x1,y1=bbox
        box=(max(0,x0-12),max(0,y0-12),min(image.width,x1+12),min(image.height,y1+12))
        x=20+424*list(EXAMPLES).index(key)
        pen.text((x,103),EXAMPLES[key],font=font(23),fill='white')
        put(image,x,142,238)
        pen.text((x,399),f'{len(gained)} additional visible blocks',font=font(20),fill='#c9fbef')
        put(overlay.crop(box),x,439,249)
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,quality=92)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    draw(args.run,args.baseline,args.out)
