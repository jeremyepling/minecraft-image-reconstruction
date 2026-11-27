"""Display a measured native screenshot fragment beside point and integrated atlas renderings."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from benchmark import view_atlas_lattice_run as run,view_atlas_lattice as experiment
from benchmark.expanded_baseline import SOURCE,frames,read,save
from benchmark.view_conditioned_atlas import footprint_uv,draw_templates
from engine.geometry import project,transform


def panel(image,size):
    image=image.copy()
    scale=min(size[0]/image.width,size[1]/image.height)
    return image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.NEAREST)


def main():
    key='test-walled-garden-1-warm-high-right'
    entry=next(e for e in frames(SOURCE,['decorative-builds-v1']) if e['id']==key)
    out=run.RUN/entry['dataset']/key
    evidence=read(out/'evidence.json.gz')
    eye=np.array(evidence['eye'])
    camera=read(out/'final/joint-camera.json')
    P=np.array(camera['P'])
    P=P/np.linalg.norm(P[2,:3])
    cat=experiment.grid.atlas()
    image=Image.open(Path(entry['frame'])/'rgb.png').convert('RGB')
    rgb=np.array(image)
    selected=None
    for a in evidence['anchors']:
        ax=a['axis']
        center=np.array(a['xyz']).mean(0)+eye
        center[ax]-=np.sign(eye[ax]-center[ax])*.02
        if a['surface']=='mask-1' and list(np.floor(center+1e-7).astype(int))==[2,0,4]:
            selected=a;break
    assert selected
    a=selected;ax=a['axis'];q=np.array(a['quad'])
    near=np.floor(q.min(0)-2).astype(int)
    far=np.ceil(q.max(0)+2).astype(int)
    xx,yy=np.meshgrid(np.arange(near[0],far[0]),np.arange(near[1],far[1]))
    native=np.column_stack((xx.ravel(),yy.ravel()))+.5
    uv=project(np.linalg.inv(transform(q)),native)
    inside=((uv>=0)&(uv<1)).all(1)
    xy=native[inside]
    orientation=a['turn']
    tile=cat.tiles[cat.labels.index(a['texture'])]
    if orientation>=4:tile=tile[:,::-1]
    tile=np.rot90(tile,orientation%4)
    observed=rgb[yy,xx].reshape(-1,3)[inside].astype('f')
    ref=[]
    for samples in (1,2):
        positions=footprint_uv(P,xy,[2,0,4],ax,samples=samples)
        ref.append(draw_templates(tile[None],positions)[0])
    gain=float((ref[1]*observed).sum()/(ref[1]**2).sum())
    panels=[]
    for data in (observed,ref[0]*gain,ref[1]*gain):
        raster=np.full((len(native),3),[28,32,36],dtype=np.uint8)
        raster[inside]=np.clip(data,0,255).round().astype('B')
        panels.append(panel(Image.fromarray(raster.reshape(xx.shape+(3,))),(356,170)))
    reference=read(run.release.ROOT/'benchmark/reviews/full-block-lattice-01/partial-example.json')
    groups=read(out/'grouped-observations.json.gz')['fragments']
    labels=read(Path(entry['frame'])/'labels.json')
    with np.load(Path(entry['frame'])/'labels.npz') as n:faces=n['face_id']
    owners=np.array([0]+[f['block'] for f in labels['faces']])
    annotated=image.copy()
    draw=ImageDraw.Draw(annotated)
    for f in groups:
        ps=f['native_pixel_indices']
        if not ps or np.mean(owners[faces.ravel()[ps]]==reference['truth']['id'])<.65:continue
        color='#56e9ad' if f['surface']=='mask-1' else '#bc92ff' if f['local']==[2,0,4] else '#ffae54'
        for p in ps:
            y,x=divmod(p,image.width)
            draw.rectangle((x-1,y-1,x+1,y+1),fill=color)
    x,y=q.mean(0).astype(int)
    img_crop=annotated.crop((x-74,y-48,x+74,y+60)).resize((370,270),Image.Resampling.NEAREST)
    scene=annotated.resize((682,426),Image.Resampling.LANCZOS)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',21)
    small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    sheet=Image.new('RGB',(1120,710),'#1d2428'); pen=ImageDraw.Draw(sheet)
    sheet.paste(scene,(15,15));sheet.paste(img_crop,(720,16))
    pen.text((720,300),'Walled Garden / held-out view',font=font,fill='#f4f3eb')
    for i,(color,line) in enumerate([('#56e9ad','Top face in cell [2, 0, 4]'),('#bc92ff','12 side-mask samples joined (11 on slab)'),('#ffae54','48 stayed one cell away (37 on slab)')]):
        pen.text((720,337+28*i),line,font=small,fill=color)
    for idx,(title,im) in enumerate(zip(('Photographed top face','Projected atlas at pixel centers','Projected atlas, four samples per pixel'),panels)):
        left=16+idx*369
        pen.text((left,465),title,font=small,fill='#f4f3eb')
        sheet.paste(im,(left,502+(175-im.height)//2))
    sheet.save(run.REVIEW/'slab-source-and-render.jpg',quality=94)
    save(run.REVIEW/'slab-render.json',dict(source_sha256=entry['rgb_sha256'],texture=a['texture'],turn=orientation,
        native_pixels=len(xy),gain=gain,point_mae=float(np.abs(ref[0]*gain-observed).mean()),
        finite_mae=float(np.abs(ref[1]*gain-observed).mean()),joint_top_anchor=a,release_true_slab=reference['truth']))


if __name__=='__main__':
    main()
