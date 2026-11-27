"""Show the native-single chart decisions on the Castle 01 source image."""
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from benchmark.local_wall_rulers import CONTROLS, WORK, inputs, read, support
from benchmark.wall_ruler_depth_ablation import CASTLE, ORIGINAL, REVIEW


def main():
    frame = next(f for f in read(CONTROLS)['pilot']['frames'] if f['id'] == CASTLE)
    row = next(r for r in read(ORIGINAL/'exports.json')['cases'] if r['id'] == CASTLE)
    image = np.asarray(Image.open(WORK/frame['rgb']['path']).convert('RGB'))
    parts, _ = inputs(frame)
    overlay = image.copy()
    for event in row['arms']['native_single']['carrier_changes']:
        part = next(p for p in parts if p['id'] == event['id'])
        mask = support(part, image.shape[:2])
        color = (255,187,64) if event['old'] else (57,208,232)
        overlay[mask > 0] = .72*image[mask > 0]+.28*np.array(color)
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,color,3)
    width = 660
    height = round(image.shape[0]*width/image.shape[1])
    sheet = Image.new('RGB',(width*2,height+58),(24,27,33))
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    pen = ImageDraw.Draw(sheet)
    pen.text((12,15),'Castle 01  |  Source image',font=font,fill='white')
    pen.text((width+12,4),'Amber   Strict chart replaced (depth -17 to +9)',font=font,fill=(255,187,64))
    pen.text((width+12,29),'Cyan      Three missing strict charts filled',font=font,fill=(57,208,232))
    for i,img in enumerate((image,overlay)):
        sheet.paste(Image.fromarray(img).resize((width,height),Image.Resampling.LANCZOS),(i*width,58))
    REVIEW.mkdir(parents=True,exist_ok=True)
    sheet.save(REVIEW/'castle-charts.jpg',quality=90)


if __name__ == '__main__':
    main()
