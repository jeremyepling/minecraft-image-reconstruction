"""Show the original image and compare the roof course in a common post-anchored grid."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.relative_canopy_spacing import shift
from benchmark.texture_camera_eval import read
from benchmark.texture_camera_sheet import font

BG='#142730'
TEXT='#f4f7f7'
GREEN='#49c49d'
BLUE='#7198ef'
RED='#f57979'


def diagram(draw,x,y,placed,reference,posts,box):
    x0,z0,x1,z1=box
    scale=min(24,340/(x1-x0+1),184/(z1-z0+1))
    left=x+(350-(x1-x0+1)*scale)/2
    top=y+(190-(z1-z0+1)*scale)/2
    for px,pz in sorted(placed|reference|posts):
        here=px,pz
        color=GREEN if here in reference and here in placed else BLUE if here in reference else RED
        a,b=left+(px-x0)*scale,top+(pz-z0)*scale
        area=(a,b,a+max(1,scale-1),b+max(1,scale-1))
        if here in placed|reference:
            draw.rectangle(area,fill=color)
        if here in posts:
            draw.rectangle(area,outline='white',width=max(1,min(3,int(scale/3))))


def render(runs,out):
    names=[name for name,_ in runs]
    data={name:{r['id']:r for r in read(path/'cases.json') if 'roof' in r} for name,path in runs}
    control=data[names[0]]
    assert all(records.keys()==control.keys() for records in data.values())
    def delta(key):
        return [data[n][key]['roof']['spacing']['anchored_roof_cells']-
                control[key]['roof']['spacing']['anchored_roof_cells'] for n in names[1:]]
    keys=['shape-benchmark-26.3--train-canopy-06-left']
    others=sorted((k for k in control if k not in keys),key=lambda k:(-max(delta(k)),k))
    keys.append(others[0])
    keys.append(min((k for k in others if k not in keys),key=lambda k:(min(delta(k)),k)))
    samples={f['id']:(split,f,prior) for split,f,prior in framed.cases(['old','fresh','extra']) if f['id'] in keys}
    img=Image.new('RGB',(1188,430+len(runs)*272),BG)
    pen=ImageDraw.Draw(img)
    def label(x,y,t,pt=19,color=TEXT):
        pen.text((x,y),t,font=font(pt),fill=color)
    label(24,16,'Roof spans measured from the front posts',34)
    label(24,73,'Green  correct roof block       Blue  missed       Red  extra at roof height       White outline  front posts',20)
    label(24,111,'Each column uses one top-down grid. The counts include only roof cells visible in that photograph.',19)
    for j,key in enumerate(keys):
        split,f,prior=samples[key]
        x=22+j*390
        rgb,_,_,_=framed.photo(split,f,prior)
        label(x,165,f['source_id'].removeprefix('train-').removeprefix('dev-').replace('-',' ').title(),25)
        thumb=ImageOps.contain(rgb,(365,207),Image.Resampling.LANCZOS)
        img.paste(thumb,(x+(365-thumb.width)//2,207))
        spec=framed.roof_specs()[key]
        truth,_,_,_=framed.references('old',f,framed.NATIVE_INPUTS[split],framed.TRAINING_DATA,framed.DEPTH,
                                      dict(old=framed.transfer.BUILDS))
        roof={(a,c) for a,b,c in truth if b==spec['roof_y']}
        posts={(spec['left']-1,spec['front_z']),(spec['right'],spec['front_z'])}
        predicted=[]
        for name,path in runs:
            row=data[name][key]['roof']['spacing']
            cells=set(map(tuple,read(path/split/key/'cells.json')['world']))
            aligned=shift(cells,row['offset']) if row['offset'] is not None else set()
            course={(a,c) for a,b,c in aligned if b==spec['roof_y']}
            assert len(course&roof)==row['anchored_roof_cells']
            predicted.append(course)
        all_xz=set.union(roof,posts,*predicted)
        box=min(a for a,c in all_xz),min(c for a,c in all_xz),max(a for a,c in all_xz),max(c for a,c in all_xz)
        for i,((name,_),course) in enumerate(zip(runs,predicted)):
            y=430+i*272
            label(x,y,name,24)
            label(x,y+34,f'{len(course&roof)} / {len(roof)} visible roof blocks',18)
            diagram(pen,x,y+68,course,roof,posts,box)
    out.parent.mkdir(parents=True,exist_ok=True)
    img.save(out,quality=93)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',nargs=2,action='append',metavar=('LABEL','PATH'),required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    render([(name,Path(path)) for name,path in a.run],a.out)
