"""Draw the projected broad face and neighboring thin face of three tested cube proposals."""
from PIL import Image,ImageDraw,ImageOps
import numpy as np

from benchmark.thin_face_placement_run import HERE,REPORT,suite,framed,read
from benchmark.thin_face_placement import project
from benchmark.texture_camera_sheet import font
from engine.pillar_seams import orientation

CASES={
    'signed-relief-26.3--dev--dev-relief-00-oblique-right':('Dev Relief 00',(5,2,-2),'A correct candidate the source checks missed'),
    'signed-relief-26.3--test--test-relief-00-oblique-left':('Relief 00',(-2,4,-2),'A correct candidate with a mostly same-axis blocker'),
    'legacy-light_gray_concrete-holdout-galleries-plaster-low-left':('Galleries',(9,2,-12),'An incorrect depth despite aligned mask edges')
}


def main():
    board=Image.new('RGB',(1272,585),'#192731')
    draw=ImageDraw.Draw(board)
    draw.text((16,12),'Does one cube explain both visible faces?',font=font(26),fill='white')
    draw.text((16,53),'Blue = counted face. Gold = thin adjoining face. White = shared edge. Box = detail shown below.',font=font(16),fill='#dfedf0')
    for job in suite.jobs(framed.EVALUATION):
        key=job[1]['id']
        if key not in CASES:continue
        name,coordinate,note=CASES[key];col=list(CASES).index(key)
        left=10+col*420
        source,_,_,camera=suite.ordinary_inputs(*job)
        P,_=orientation(camera,source.size);eye=np.linalg.solve(P[:,:3],-P[:,3])
        candidates=read(HERE/'runs/thin-final'/(key+'.json'))['rows']
        proposal=next(r for r in candidates if tuple(r['cell'])==coordinate)
        points=[]
        overlay=source.copy();mark=ImageDraw.Draw(overlay)
        for role,color in [('host','#47caff'),('side','#ffbd4e')]:
            axis=proposal[role+'_axis'];sign=eye[axis]>coordinate[axis]+1
            q=np.tile(np.array(coordinate,dtype=float),(4,1));q[:,axis]+=int(sign)
            q[:,[i for i in range(3) if i!=axis]] +=[[0,0],[1,0],[1,1],[0,1]]
            xy,_=project(q,P);xy=list(map(tuple,xy));points.extend(xy)
            mark.line(xy+[xy[0]],fill=color,width=2)
        mark.line(list(map(tuple,proposal['edge'])),fill='white',width=2)
        center=np.mean(points,axis=0);x,y=map(int,center)
        box=(x-78,y-56,x+78,y+56)
        full=source.copy()
        ImageDraw.Draw(full).rectangle(box,outline='#ffdd87',width=5)
        thumb=ImageOps.contain(full,(405,230))
        board.paste(thumb,(left+(405-thumb.width)//2,111))
        zoom=ImageOps.contain(overlay.crop(box),(360,183),Image.Resampling.NEAREST)
        board.paste(zoom,(left+(405-zoom.width)//2,354))
        draw.text((left,81),name,font=font(20),fill='white')
        draw.text((left,550),note,font=font(14),fill='#e9eef0')
    board.save(REPORT/'corner-examples.jpg',quality=92)


if __name__=='__main__':
    main()
