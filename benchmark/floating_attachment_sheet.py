"""Show every offered detached chart on its own source crop and under the measured registration."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from benchmark.floating_expanded_wall_attachment import (REVIEW,baseline,permitted,read,workload,axis_vector,oriented)
from benchmark.relative_face_scoring import _yaw
from engine.geometry import UNIT,project


def main():
    cases={(g,c['id']):(c,row) for g,c,row,_ in workload()}
    rows=[r for r in read(REVIEW/'conditions.json.gz') if r['attachments']]
    canvas=Image.new('RGB',(1120,325*math.ceil(len(rows)/2)),'white')
    draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        g,n=row['panel'],row['id']
        case,archived=cases[g,n]
        folder,_=baseline(g,n,archived)
        image=Image.open(case['variants']['sky']['image']).convert('RGB')
        charts,_=oriented(read(folder/'charts/sky'/n/'charts.json')['charts'])
        by={c['id']:permitted(c) for c in charts}
        labels=read(Path(case['frame'])/'labels.json')
        blocks={b['id']:b['block_id'] for b in labels['blocks']}
        truth={tuple(f['xyz_world']):blocks[f['block']] for f in labels['faces']}
        overlay=ImageDraw.Draw(image)
        for a in row['attachments']:
            p,c=by[a['parent']],by[a['child']]
            u,v=map(axis_vector,p['axes'])
            normal=axis_vector(p['normal_axis'])
            origin=np.asarray(row['selected_origins'][p['id']])+np.minimum(0,u)+np.minimum(0,v)-np.maximum(0,normal)
            fixed=row['score']['tangent']
            points=[]
            for cell in c['cells']:
                uv=np.array([cell['u'],cell['v']])
                xy=project(np.asarray(c['H']),uv+UNIT)
                points.extend(xy)
                u1,v1=uv+a['fit']['shift']
                xyz=tuple(np.asarray(_yaw(origin+u1*u+v1*v,fixed['yaw_quarters']))+fixed['translation'])
                color='#00ff91' if xyz in truth else '#ff4949'
                overlay.line([tuple(t) for t in [*xy,xy[0]]],fill=color,width=3)
        points=np.array(points)
        dx,dy=np.maximum(np.ptp(points,axis=0)*.35,40.)
        lo=np.maximum(points.min(0)-[dx,dy],0)
        hi=np.minimum(points.max(0)+[dx,dy],image.size)
        cut=image.crop(tuple(map(int,(*lo,*hi))))
        cut.thumbnail((545,235),Image.Resampling.LANCZOS)
        x=560*(i%2);y=325*(i//2)
        canvas.paste(cut,(x+(560-cut.width)//2,y+43))
        draw.text((x+8,y+6),n,fill='black',font=ImageFont.truetype('arial.ttf',17))
        d=row['paired']['tangent']
        text=f"Tangent fit: {d['gained']} correct, {d['added']} extra"
        if row['attachments'][0]['same_partition']:
            text+=' (blocked by partition guard)'
        draw.text((x+8,y+286),text,fill='black',font=ImageFont.truetype('arial.ttf',15))
    canvas.save(REVIEW/'detached-source-charts.jpg',quality=91)


if __name__=='__main__':
    main()
