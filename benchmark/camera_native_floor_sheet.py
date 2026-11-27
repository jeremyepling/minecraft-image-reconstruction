"""Render source RGB, the independent floor checks and native projected corners."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.camera_native_floor_run import REVIEW, SOURCE, read
from benchmark.camera_metrics import registration


def font(size):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except OSError:
        return ImageFont.load_default()


def main():
    cases={r['id']:r for r in read(REVIEW/'native-cameras.json.gz')}
    names=['train-pillars-02-marble-low-left','validation-harbor-02-seafoam-rear-oblique',
           'test-castle-01-ashlar-high-right','test-castle-03-ashlar-rear-oblique']
    out=Image.new('RGB',(1480,1132),'#161b20')
    draw=ImageDraw.Draw(out)
    draw.text((20,12),'Native floor verification and visible-corner reprojection',font=font(26),fill='white')
    draw.text((20,51),'Cyan marks fit regions. Amber marks regions excluded from the final fit. Magenta shows reference corners projected with the estimated camera.',font=font(17),fill='#dae1e6')
    for index,key in enumerate(names):
        row=cases[key]
        frame=SOURCE/row['dataset']/'data'/row['split']/'frames'/key
        image=Image.open(frame/'rgb.png').convert('RGB')
        d=ImageDraw.Draw(image);w,h=image.size
        for lo,hi,color in ((.04,.34,'cyan'),(.89,.99,'cyan'),(.40,.54,'orange'),(.61,.84,'orange')):
            d.rectangle((int(lo*w),int(.79*h),int(hi*w),int(.98*h)),outline=color,width=3)
        if 'relative' in row:
            P=np.array(row['camera']['P'])
            aligned=P@np.linalg.inv(registration(P,read(frame/'camera.json')))
            corners={tuple(xyz) for face in read(frame/'labels.json')['faces']
                     for xyz,ok in zip(face['corners_world'],face['corner_visible']) if ok}
            pts=(np.array(sorted(corners))[1::2]+row['relative']['integer_world_xyz_offset_from_registered'])
            pts=pts[::max(1,len(pts)//65)]
            uv=pts@aligned[:,:3].T+aligned[:,3]
            for x,y in uv[:,:2]/uv[:,2:]:
                d.line((x-4,y,x+4,y),fill='#ff37c5',width=2);d.line((x,y-4,x,y+4),fill='#ff37c5',width=2)
            status=f"{row['choice']}, P90 native error {row['relative']['p90_check_px']:.2f} px"
        else:
            status='Abstained. The floor patches did not jointly verify the chart.'
        x=20+740*(index%2);y=89+519*(index//2)
        out.paste(image.resize((720,450),Image.Resampling.LANCZOS),(x,y))

        draw.text((x+4,y+455),key,font=font(17),fill='white')
        draw.text((x+4,y+481),status,font=font(16),fill='#e4e8ed')
    out.save(REVIEW/'native-source-examples.jpg',quality=91)


if __name__=='__main__':main()
