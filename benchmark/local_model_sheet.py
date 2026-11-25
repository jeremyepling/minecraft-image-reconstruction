"""Summarize three optional local probes on the identical native development screenshot."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import read, save
from benchmark.depth_probe_score import occlusion_pairs
from benchmark.surface_metrics import DOWNSAMPLE, truth
from benchmark.surface_sheet import PAPER, DIM, panel, scene_box, text
from engine.geometry import bilinear


def make(fixture, line_dir, depth_dir, normal_dir, out):
    lines = read(line_dir/'segments.json')
    depth = read(depth_dir/'probe.json')
    normals = read(normal_dir/'report.json')
    key = lines['image_sha256']
    if key!=depth['source_sha256'] or key!=normals['image_sha256']:
        raise ValueError('All model previews must use the same frame')
    frame, = [f for f in read(fixture/'index.json')['frames'] if f['image_sha256']==key]
    if frame['split'] != 'dev':
        raise ValueError('This research preview uses development frames.')
    gt = truth(read(fixture/frame['pose']), read(fixture/frame['labels'])['faces'])
    src = Image.open(fixture/frame['image']).convert('RGB')
    box = scene_box(gt,src)
    ids = gt['id_map']
    inside = ids>0
    yy,xx = np.where(inside)
    normal = np.load(normal_dir/'normals.npz',allow_pickle=False)['normal']
    samples = bilinear(normal,(np.column_stack((xx,yy))+.5)*DOWNSAMPLE*np.array(normal.shape[1::-1])/src.size)
    expected = np.array([[0,0,0]]+[gt['patches'][i]['normal_camera'] for i in sorted(gt['patches'])])[ids[inside]]
    angles = np.degrees(np.arccos(np.clip(((samples/np.linalg.norm(samples,axis=1,keepdims=True))*expected).sum(1),-1,1)))
    truth_rgb = np.full((*ids.shape,3),47,dtype=np.uint8)
    truth_rgb[inside] = np.uint8(np.clip(255*(.5+expected*[.5,-.5,-.5]),0,255))
    error = np.full((*ids.shape,3),47,dtype=np.uint8)
    stops = np.array([[34,154,125],[248,201,101],[215,85,100]])
    error[inside] = np.stack([np.interp(angles,[0,15,45],col) for col in stops.T],axis=-1).astype(np.uint8)
    relative = np.load(depth_dir/'relative-depth.npy',allow_pickle=False)
    # Local contrast is labeled: map the building crop's own 1st/99th percentile to the display range.
    depth_map = np.array(Image.fromarray(relative).resize(src.size,Image.Resampling.BILINEAR))
    low,high = np.percentile(depth_map[box[1]:box[3],box[0]:box[2]],(1,99))
    depth_rgb = Image.fromarray(np.uint8(np.clip(255*(depth_map-low)/max(high-low,1.e-6),0,255))).convert('RGB')
    scores = occlusion_pairs(gt,relative)
    line_score = read(line_dir/'boundary-score.json')
    sheet = Image.new('RGB',(1550,900),PAPER)
    draw = ImageDraw.Draw(sheet)
    text(draw,30,20,'Three local vision models on the same shopfront',37,True)
    text(draw,30,78,'One development view   /   vanilla Java 26.3   /   clear noon   /   light gray concrete floor',21,fill=DIM)
    panels = [
        ('Source screenshot',src,'Recesses, posts and the awning in one view.'),
        ('M-LSD Tiny  ·  line segments',Image.open(line_dir/'lines.png'),
         f"{lines['segment_count']} segments across the full frame."),
        ('Depth Anything V2 Small  ·  relative depth',depth_rgb,'Brighter is nearer. Contrast scaled within this crop.'),
        ('MoGe-2 Small  ·  predicted normals',Image.open(normal_dir/'normal.png'),'Color encodes the direction a surface faces.'),
        ('Recorded normals',Image.fromarray(truth_rgb),'World geometry, scored on visible building pixels.'),
        ('Error in predicted orientation',Image.fromarray(error),'Green 0°    /    amber 15°    /    rose 45° or more.')
    ]
    for i,(title,im,desc) in enumerate(panels):
        x,y=30+(i%3)*500,140+(i//3)*365
        pts = np.array(box).reshape(2,2)*np.array(im.size)/src.size
        b=tuple(np.rint(pts).astype(int).ravel())
        text(draw,x,y,title,21,True)
        panel(sheet,im.convert('RGB'),b,x,y+35,483,256)
        text(draw,x,y+305,desc,15)
    out.mkdir(parents=True,exist_ok=True)
    sheet.save(out/'preview.jpg',quality=94,subsampling=0)
    report = dict(kind='Single development-frame diagnostic',image=frame['id'],image_sha256=key,
                  floor=frame['floor'],region_xyxy=box,artifact='preview.jpg',
                  mlsd={k:v for k,v in lines.items() if k!='segments'},physical_boundaries=line_score,
                  relative_depth={k:v for k,v in depth.items() if k!='artifacts'},
                  depth_ordering=scores,normal={k:v for k,v in normals.items() if k!='artifacts'})
    save(out/'measurements.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','lines','depth','normals','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    make(args.fixture,args.lines,args.depth,args.normals,args.out)
