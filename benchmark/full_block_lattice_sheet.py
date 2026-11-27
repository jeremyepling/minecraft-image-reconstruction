"""Generate two evidence overlays from the frozen source footprints and their independent grader."""
from collections import Counter,defaultdict
from io import BytesIO
from pathlib import Path
from shutil import copyfile
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from benchmark import full_block_lattice_run as run
from benchmark.expanded_baseline import SOURCE,frames,read,save


def main():
    ids=('train-market-court-1-cool-low-left','test-walled-garden-1-warm-high-right')
    for entry in frames(SOURCE,['decorative-builds-v1']):
        if entry['id'] not in ids:
            continue
        key=entry['id']
        out=run.RUN/entry['dataset']/key
        source=Path(entry['frame'])
        evidence=read(out/'evidence.json.gz')
        grouped=read(out/'grouped-observations.json.gz')
        photo=Image.open(source/'rgb.png').convert('RGB')
        pen=ImageDraw.Draw(photo)
        for a in evidence.get('anchors',[]):
            p=[tuple(q) for q in a['quad']]
            pen.line(p+p[:1],fill='#35df88',width=2)
        if key==ids[0]:
            for f in grouped['fragments']:
                for p in f['native_pixel_indices'][::2]:
                    y,x=divmod(p,photo.width)
                    pen.point((x,y),fill='#ffb246')
            photo.thumbnail((1080,675))
            photo.save(run.REVIEW/'floor-rejection.jpg',quality=90)
            copyfile(out/'proposal.zip',run.REVIEW/'market-court-source.zip')
            continue
        labels=read(source/'labels.json')
        owners=np.array([0]+[f['block'] for f in labels['faces']])
        with np.load(source/'labels.npz') as data:
            raster=data['face_id']
        by=defaultdict(list)
        for f in grouped['fragments']:
            ps=f['native_pixel_indices']
            if not ps: continue
            front=raster.ravel()[ps]
            o,n=Counter(owners[front].tolist()).most_common(1)[0]
            if not o or n/len(ps)<.65 or labels['blocks'][o-1].get('geometry_kind')=='full_cube': continue
            face=Counter(front[owners[front]==o].tolist()).most_common(1)[0][0]
            by[o].append((face,f))
        pairs=[]
        from itertools import combinations
        for o,g in by.items():
            for (a,f),(b,h) in combinations(g,2):
                if a!=b and f['surface']!=h['surface']:
                    pairs.append((min(len(f['native_pixel_indices']),len(h['native_pixel_indices'])),o,a,b,f,h))
        _,o,a,b,f,h=max(pairs,key=lambda row:row[0])
        pts=[]
        for piece,color in ((f,'#ffb544'),(h,'#20c9ff')):
            for p in piece['native_pixel_indices']:
                y,x=divmod(p,photo.width); pts.append((x,y))
                pen.rectangle((x-1,y-1,x+1,y+1),fill=color)
        x,y=np.array(pts).mean(0).astype(int)
        zoom=photo.crop((x-74,y-58,x+74,y+58)).resize((444,348),Image.Resampling.NEAREST)
        photo.thumbnail((1000,625))
        sheet=Image.new('RGB',(1000,1003),'#202521')
        sheet.paste(photo,(0,0)); sheet.paste(zoom,(10,644))
        d=ImageDraw.Draw(sheet)
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',22)
        desc=labels['blocks'][o-1]
        lines=['Two observed subfaces of one '+desc['geometry_kind'],
               'Orange cell '+str(f['local']),'Blue cell '+str(h['local']),
               'Reference '+str(desc['xyz_world']),'The counter kept two occupied cells.']
        for i,s in enumerate(lines): d.text((474,675+i*48),s,fill='#faf8ed',font=font)
        sheet.save(run.REVIEW/'partial-offset.jpg',quality=92)
        save(run.REVIEW/'partial-example.json',dict(id=key,source_sha256=entry['rgb_sha256'],
            truth=desc,predicted=[dict(surface=p['surface'],local=p['local'],planes=p['planes']) for p in (f,h)],
            true_faces=[int(a),int(b)]))


if __name__=='__main__':
    main()
