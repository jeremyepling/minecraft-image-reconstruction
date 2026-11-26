"""Draw measured cameras and decoded visible placement on three fixed screenshots."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.camera_metrics import registration, truth_camera
from engine.geometry import UNIT, project


CASES = [('validation','validation-cloister1-cool-high-right','Cloister'),
         ('test','test-twinspire1-cool-high-right','Twin Spires'),
         ('regression','legacy-light_gray_concrete-holdout-galleries-plaster-high-right','Galleries')]


def read(path):
    return json.loads(path.read_bytes())


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def mixed_sheet(run, out):
    key = 'train-arcbridge1-cool-high-right'
    row = next(r for r in read(run/'control.json')['results'] if r['id']==key and r['palette']=='distinct')
    P = np.asarray(row['P'])
    def quad(xz):
        return project(P,np.insert(UNIT+xz,1,0.,axis=1))
    with Image.open(run/f'{key}-distinct.png') as im:
        image = im.convert('RGB')
    center = np.array([.5*image.width,.84*image.height])
    edge = min(row['atlas_edges'],key=lambda e:np.linalg.norm(quad(e['first']).mean(0)-center))
    boxes = [quad(edge[k]) for k in ('first','second')]
    all_corners = np.concatenate(boxes)
    bounds = (max(0,int(all_corners[:,0].min()-35)),max(0,int(all_corners[:,1].min()-22)),
              min(image.width,int(all_corners[:,0].max()+35)),min(image.height,int(all_corners[:,1].max()+22)))
    colors = ['#18eaff','#ffb14b']
    sheet = Image.new('RGB',(1040,414),'#18212b')
    pen = ImageDraw.Draw(sheet)
    pen.text((14,12),'Locating a shared block edge from two different atlas textures',font=font(24))
    scene = image.copy()
    draw = ImageDraw.Draw(scene)
    for q,col in zip(boxes,colors):
        draw.line([tuple(x) for x in [*q,q[0]]],fill=col,width=3)
    sheet.paste(scene.resize((544,340)),(14,58))
    crop = scene.crop(bounds)
    factor = min(455/crop.width,220/crop.height)
    crop = crop.resize((round(factor*crop.width),round(factor*crop.height)),Image.Resampling.NEAREST)
    sheet.paste(crop,(572,64))
    for n,(name,col) in enumerate(zip(edge['predicted'],colors)):
        pen.text((572,293+27*n),name.replace('_',' '),font=font(21),fill=col)
    pen.text((572,362),'Generated mixed floor, held-out control view',font=font(16))
    sheet.save(out,quality=91)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--placement',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mixed',type=Path,help='Also illustrate the check/high-right mixed-material run.')
    a = p.parse_args()
    tile = (400,240)
    sheet = Image.new('RGB',(1250,850),'#18212b')
    draw = ImageDraw.Draw(sheet)
    draw.text((16,7),'Where the recovered camera helps, and where placement still misses',font=font(24))
    draw.text((16,40),'Camera row: teal crosses = prediction, white dots = reference.  Placement row: green = correct, pink = missed, orange = extra.',font=font(17))
    for col,(split,key,title) in enumerate(CASES):
        d = a.data/split/'frames'/key
        with Image.open(d/'rgb.png') as im:
            img = im.convert('RGB')
        faces = read(d/'reference-faces.json')['faces']
        pose = read(d/'camera.json')
        row = next(r for r in read(a.placement/split/'cases.json') if r['id']==key)
        recovered = np.asarray(row['status']['P'])
        P = recovered @ np.linalg.inv(registration(recovered,pose))
        ref = truth_camera(pose)
        q = np.vstack([f['image_corners'] for f in faces])
        x0,y0 = np.floor(q.min(0)-15).astype(int)
        x1,y1 = np.ceil(q.max(0)+15).astype(int)
        crop = (max(0,x0),max(0,y0),min(img.width,x1),min(img.height,y1))
        world = read(a.placement/split/key/'registered.json')
        correct = set(map(tuple,world['correct']))
        extra = set(map(tuple,world['registered']))-correct
        placement = img.copy()
        pen = ImageDraw.Draw(placement,'RGBA')
        for f in faces:
            point = tuple(f['xyz'])
            center = np.mean(f['image_corners'],axis=0)
            x,y = center
            pen.ellipse((x-2,y-2,x+2,y+2),fill=(90,255,144,210) if point in correct else (255,95,174,200))
        if extra:
            xyz = np.asarray(sorted(extra))+.5
            front = xyz @ ref[2,:3]+ref[2,3] > 0
            for x,y in project(ref,xyz[front]):
                pen.line((x-3,y-3,x+3,y+3),fill=(255,155,50,240),width=2)
        camera = img.copy()
        pen = ImageDraw.Draw(camera)
        vertices = {tuple(pt):xy for f in faces for pt,xy,visible in
                    zip(f['world_corners'],f['image_corners'],f['corner_visible']) if visible}
        for (x,y),(u,v) in zip(project(P,list(vertices)),vertices.values()):
            pen.line((x-3,y,x+3,y),fill='#26efea',width=2)
            pen.line((x,y-3,x,y+3),fill='#26efea',width=2)
            pen.point((u,v),fill='white')
        left=12+col*414
        draw.text((left,70),title,font=font(20))
        for n,im in enumerate((img,camera.crop(crop),placement.crop(crop))):
            im.thumbnail(tile)
            sheet.paste(im,(left+(tile[0]-im.width)//2,98+n*240+(tile[1]-im.height)//2))
        s=row['placement']
        draw.text((left,823),f"{s['correct_visible']} / {s['visible_truth']} correct, {s['extra']} extra",font=font(17))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(a.out,quality=91)
    if a.mixed:
        mixed_sheet(a.mixed,a.out.with_name('material-joint.jpg'))


if __name__=='__main__':
    main()
