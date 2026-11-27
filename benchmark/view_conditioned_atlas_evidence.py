"""Render auditable source/sprite examples and material results by estimated view geometry."""
from collections import defaultdict
from io import BytesIO
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from benchmark.view_conditioned_atlas_run import BASE,OUT,REVIEW,inputs,read,save,_export_atlas,decoded_pack
from benchmark.view_conditioned_atlas import footprint_uv,draw_templates
from engine.projected_appearance import Scene
from engine.placed_materials import Catalog
from engine.thin_atlas_faces import _corr

def measure(rows):
    grouped=defaultdict(lambda:[0,0,0,0])
    candidates=defaultdict(list)
    for row in rows:
        ds,key=row['dataset'],row['id']
        details=read(OUT/'details'/ds/(key+'.json.gz'))
        if not details:continue
        rgb,camera,pack,_=inputs(row)
        P=np.asarray(camera['P']);eye=-np.linalg.solve(P[:,:3],P[:,3])
        for d in details:
            if not d['truth']:continue
            center=np.array(d['xyz'])+.5;axis=d['axis']
            center[axis]+= .5 if eye[axis]>center[axis] else -.5
            offset=eye-center
            distance=float(np.linalg.norm(offset))
            angle=float(np.rad2deg(np.arccos(np.clip(abs(offset[axis])/distance,0,1))))
            truth=d['truth'].split(':')[-1]
            values=[1]+[int(d[k]==truth) for k in ('old','static','view')]
            for field in [('distance', 'near <20' if distance<20 else 'mid 20-40' if distance<40 else 'far >=40'),
                          ('angle','frontal <30' if angle<30 else 'moderate 30-60' if angle<60 else 'oblique >=60')]:
                group=grouped[(ds,*field)]
                for i,val in enumerate(values):group[i]+=val
            if d['pixels']>=100:
                outcome=('gain' if values[3] and not values[2] else 'loss' if values[2] and not values[3] else None)
                if outcome:
                    candidates[outcome].append((angle,distance,d['pixels'],row,d))
    save(REVIEW/'view-bins.json',[dict(dataset=k[0],measure=k[1],bin=k[2],supported_aligned=v[0],baseline=v[1],point=v[2],source=v[3]) for k,v in grouped.items()])
    # Purposefully contrasting geometry. Pick large samples in a distant oblique
    # gain and a near frontal loss when available. These are illustrative.
    picks=[]
    for mode in ['gain','loss']:
        pool=candidates[mode]
        if mode=='gain':
            sub=[r for r in pool if r[0]>=55 and r[1]>=30]
        else:
            sub=[r for r in pool if r[0]<35 and r[1]<25]
        picks.append(max(sub or pool,key=lambda r:r[2]))
    return picks

def panel(row,d):
    rgb,camera,pack,_=inputs(row)
    photo=Image.open(rgb).convert('RGB')
    ids=decoded_pack(BytesIO(pack))
    scene=Scene(photo,[],camera,ids,step=max(1,(max(photo.size)+1599)//1600))
    cell=tuple(d['xyz']);axis=d['axis'];n=scene.cells.index(cell)
    idx=scene.owned[n];idx=idx[scene.axes[idx]==axis]
    y,x=np.divmod(idx,scene.shape[1])
    xy=np.column_stack((x,y))*scene.step+.5
    uv=footprint_uv(scene.P,xy,cell,axis,samples=3)
    centers=scene.uv(idx,cell,axis)
    pix=scene.rgb[y,x]
    bounds=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    width,height=bounds[2]-bounds[0],bounds[3]-bounds[1]
    face=np.zeros((height,width,3),dtype=np.uint8)+28
    atlas=_export_atlas();cat=Catalog(atlas)
    face[y-y.min(),x-x.min()]=pix
    overview=photo.copy()
    marker=ImageDraw.Draw(overview)
    marker.rectangle(tuple(v*scene.step for v in bounds),outline='#ffbd30',width=4)
    panels=[overview,photo.crop(((bounds[0]-width//2)*scene.step,(bounds[1]-height//2)*scene.step,
                        (bounds[2]+width//2)*scene.step,(bounds[3]+height//2)*scene.step)),Image.fromarray(face.copy())]
    side=('top' if scene.eye[1]>cell[1]+1 else 'bottom') if axis==1 else 'wall'
    for arm,sampling in [('static',centers[:,None]),('view',uv)]:
        valid=cat.valid[side]&(cat.candidates==cat.blocks.index(d[arm]))
        tex=np.unique(cat.sprites[valid])
        tiles=np.stack([np.rot90(arr,k,axes=(1,2)) for arr in (atlas.tiles[tex],atlas.tiles[tex,:,::-1]) for k in range(4)],axis=1)
        rend=draw_templates(tiles.reshape(-1,16,16,3),sampling)
        winner=_corr(rend.mean(2),pix.mean(1)).argmax()
        pred=rend[winner]
        gain=np.clip((pred.mean(0)@pix.mean(0))/max(1.,pred.mean(0)@pred.mean(0)),.26,1.08)
        face[y-y.min(),x-x.min()]=np.clip(gain*pred,0,255).astype(np.uint8)
        panels.append(Image.fromarray(face.copy()))
    return panels

def main():
    rows=read(BASE/'scores/cases.json')
    selected=measure(rows)
    out=Image.new('RGB',(1300,520),'#f4f3ef')
    draw=ImageDraw.Draw(out)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    info=[]
    for i,(angle,distance,pixels,row,d) in enumerate(selected):
        y=i*260
        label='Footprint recovers the correct ID' if i==0 else 'Point sampler retains the correct ID'
        draw.text((12,y+2),label+'  |  '+row['id'],font=font,fill='#182922')
        labels=['Source image','Source context','Observed face','Point: '+d['static'],'Footprint: '+d['view']]
        for j,(im,title) in enumerate(zip(panel(row,d),labels)):
            im.thumbnail((245,145)) if im.width>245 or im.height>145 else None
            ratio=min(245/im.width,145/im.height)
            im=im.resize((max(1,round(im.width*ratio)),max(1,round(im.height*ratio))),Image.Resampling.NEAREST)
            x=j*260
            out.paste(im,(x+(260-im.width)//2,y+45+(145-im.height)//2))
            draw.text((x+8,y+192),title,font=font,fill='#182922')
        draw.text((12,y+222),f"Recorded ID {d['truth']}    Estimated angle {angle:.1f} degrees    Estimated distance {distance:.1f} blocks",font=font,fill='#324d42')
        info.append(dict(id=row['id'],dataset=row['dataset'],source_sha256=row['source_sha256'],base_pack_sha256=row['pack_sha256'],cell=d,
                         estimated_angle_degrees=angle,estimated_distance_blocks=distance))
    out.save(REVIEW/'source-and-rendered-faces.png')
    save(REVIEW/'example-sources.json',info)

if __name__=='__main__':main()
