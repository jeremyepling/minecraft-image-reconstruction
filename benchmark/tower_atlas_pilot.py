"""One-frame, screenshot-seeded atlas/scale probe on the projecting tower facade."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Coordinates of six visible window-head corners, read from the supplied frame.
FIT = [(0,0,371,201),(2,0,407,204),(6,0,480,210),
       (0,5,371,296.5),(2,5,407.5,299),(6,5,480,304)]
# Test seams are corners on different rows, below all fitting points.
CHECK = [(0,8,371,354),(3,8,426,357),(6,8,480,361),
         (0,10,371,393),(3,10,426,396),(6,10,480,400),
         (0,12,371,433.5),(6,12,480,438)]
# Manually selected visibly textured full-face candidates, excluding recesses/trim.
CELLS = [(0,2),(0,3),(2,2),(2,3),(2,4),(4,2),(6,3),(5,0),
         (1,10),(2,10),(6,11),(4,9),(7,8),(1,-2)]
SCALES = (.5,1.,2.)

def transform(points, matrix):
    h = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return h[:,:2] / h[:,2:3]

def homography():
    rows = []
    for u,v,x,y in FIT:
        rows.extend(([u,v,1,0,0,0,-u*x,-v*x,-x],
                     [0,0,0,u,v,1,-u*y,-v*y,-y]))
    return np.linalg.svd(rows)[2][-1].reshape(3,3)

def sample(im, xy):
    x,y = xy.T
    i,j = np.floor(x).astype(int),np.floor(y).astype(int)
    dx,dy = (x-i)[:,None],(y-j)[:,None]
    return ((1-dy)*((1-dx)*im[j,i]+dx*im[j,i+1])
            +dy*((1-dx)*im[j+1,i]+dx*im[j+1,i+1])).astype('f')

def atlas(directory):
    meta = json.loads((directory/'atlas.json').read_text())
    sheet = Image.open(directory/'atlas.png').convert('RGBA')
    tiles, names = [], []
    for entry in meta['textures']:
        if not entry['face_blocks'] or entry['source_size'] != [16,16]:
            continue
        x,y,w,h = entry['rect']
        tile = sheet.crop((x,y,x+w,y+h)).resize((16,16),Image.Resampling.BOX)
        raw = np.asarray(tile)
        if raw[:,:,3].min() < 255:
            continue
        for k in range(4):
            tiles.append(np.rot90(raw[:,:,:3], k))
            names.append((entry['id'],k))
    return np.array(tiles), names, meta['atlas_sha256']

def observation(im, H, cell, mode):
    if mode.startswith('rectified'):
        n = 32 if mode == 'rectified32' else 16
        t = (np.arange(n)+.5)/n
        uv = np.stack(np.meshgrid(t,t),axis=-1).reshape(-1,2)
        return uv, sample(im, transform(uv+cell,H))
    corners = transform(np.array([[0,0],[1,0],[1,1],[0,1]])+cell,H)
    x,y = np.floor(corners.min(0)).astype(int)
    X,Y = np.ceil(corners.max(0)).astype(int)
    xy = np.stack(np.meshgrid(np.arange(x,X+1), np.arange(y,Y+1)),axis=-1).reshape(-1,2)
    uv = transform(xy,np.linalg.inv(H))-cell
    keep = ((uv>=0)&(uv<1)).all(1)
    return uv[keep], im[xy[keep,1],xy[keep,0]]

def match(tiles,names,uv,target,scale):
    # Compare a constant footprint. Half pitch repeats within it. Double pitch
    # tries each quadrant, because the marked face can lie in any quadrant.
    phases = [(0,0)] if scale != 2 else [(0,0),(.5,0),(0,.5),(.5,.5)]
    target_mean = target.mean(0)
    yc = target-target_mean
    best = (1e6,None)
    for phase in phases:
        grid = ((uv/scale + phase)%1)*16-.5
        ix,iy = np.floor(grid).astype(int).T
        fx,fy = (grid%1).astype('f').T
        for start in range(0,len(tiles),96):
            subset = tiles[start:start+96]
            tex = np.zeros((len(subset),len(uv),3),dtype='f')
            for dx,wx in [(0,1-fx),(1,fx)]:
                for dy,wy in [(0,1-fy),(1,fy)]:
                    tex += subset[:,(iy+dy).clip(0,15),(ix+dx).clip(0,15)]*(wx*wy)[None,:,None]
            mean = tex.mean(1)
            xc = tex-mean[:,None]
            gain = ((tex*target).mean((1,2))/(tex*tex).mean((1,2)).clip(1)).clip(.65,1.55)
            # Mild warm/cool color cast and haze allowed as bounded channel offsets.
            bias = (target_mean-mean*gain[:,None]).clip(-18,18)
            mean_err = mean*gain[:,None]+bias-target_mean
            residual = xc*gain[:,None,None]-yc
            errs = np.sqrt((residual**2).mean((1,2))+(mean_err**2).mean(1))
            k = int(errs.argmin())
            if errs[k] < best[0]:
                name,rotation = names[start+k]
                best = (float(errs[k]),dict(texture=name,quarter_turns=rotation,
                        rmse=float(errs[k]),gain=float(gain[k]),phase=phase))
    return best[1]

def main():
    parser = argparse.ArgumentParser()
    for arg in ('source','atlas','out'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    im=Image.open(args.source).convert('RGB')
    rgb=np.asarray(im,dtype='f')
    tiles,names,digest=atlas(args.atlas)
    H=homography()
    overlay=im.copy()
    draw=ImageDraw.Draw(overlay)
    project=lambda pts: [tuple(point) for point in transform(pts,H)]
    for x in range(9):
        draw.line(project([(x,-2),(x,12)]),fill='#00dbff',width=1)
    for y in range(-2,13):
        draw.line(project([(0,y),(8,y)]),fill='#00dbff',width=1)
    for x,y in CELLS:
        draw.line(project([(x,y),(x+1,y),(x+1,y+1),(x,y+1),(x,y)]),fill='#ffa500',width=1)
    seams=[]
    for u,v,x,y in CHECK:
        p=transform([(u,v)],H)[0]
        seams.append(float(np.linalg.norm(p-[x,y])))
        draw.ellipse((x-2,y-2,x+2,y+2),fill='#ff00dd')
    overlay.crop((330,145,530,450)).resize((400,610)).save(args.out/'grid.png')
    rows=[]
    for cell in CELLS:
        row={'cell':cell}
        for mode in ('native','rectified','rectified32'):
            uv,target=observation(rgb,H,cell,mode)
            row[mode]={s:match(tiles,names,uv,target,s) for s in SCALES}
        rows.append(row)
    # Synthetic model check: render one independent, clean known tile through
    # the same estimated plane. PIL bilinear interpolation and a warm exposure
    # are used to generate this control; the atlas search still sees all IDs.
    known='minecraft:block/dead_brain_coral_block'
    cell=np.array((2,3))
    texture=tiles[names.index((known,0))]
    control=np.zeros_like(rgb)
    yy,xx=np.indices(control.shape[:2])
    xy=np.stack((xx.ravel(),yy.ravel()),axis=1)
    uv=(transform(xy,np.linalg.inv(H))-cell)%1
    tile=Image.fromarray(texture).resize((256,256),Image.Resampling.BILINEAR)
    pix=np.clip((uv*256).astype(int),0,255)
    control[:]=np.clip(np.asarray(tile)[pix[:,1],pix[:,0]].reshape(control.shape)*1.10+[7,3,0],0,255)
    positives={}
    for mode in ('native','rectified','rectified32'):
        uv,target=observation(control,H,cell,mode)
        positives[mode]={s:match(tiles,names,uv,target,s) for s in SCALES}
    report=dict(known_control_id=known,control=positives,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),atlas_sha256=digest,
                fit=FIT,check=CHECK,seam_errors_px=seams,homography=H.tolist(),
                distinct_textures=len(tiles)//4,rows=rows)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    for mode in ('native','rectified','rectified32'):
        print(mode,'scale wins', [sum(min(r[mode],key=lambda s:r[mode][s]['rmse'])==s for r in rows) for s in SCALES],
              'median RMSE', [round(float(np.median([r[mode][s]['rmse'] for r in rows])),2) for s in SCALES])
    print('control',[(m,min(rs,key=lambda s:rs[s]['rmse']),rs[1.0]['texture']) for m,rs in positives.items()]); print('held-out seam errors',np.round(seams,1))

if __name__=='__main__':
    main()
