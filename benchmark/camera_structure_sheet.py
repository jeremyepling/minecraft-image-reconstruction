"""Show observed structural directions and the measured floor-contact segment on two native sources."""
from benchmark.camera_structure_run import REPORT,COHORTS,SOURCE,frames,paths,read,save
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path


def main():
    examples={'test-castle-02-ashlar-rear-oblique':'Castle 02, held-out rear view',
              'train-pillars-01-basalt-low-left':'Pillars 01, training low view'}
    sheet=Image.new('RGB',(1200,485),'#20242b')
    pen=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    heading=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',18)
    entries={e['id']:e for e in frames(SOURCE,COHORTS)}
    data=[]
    for i,(key,label) in enumerate(examples.items()):
        entry=entries[key]
        evidence=read(paths(entry)[1]/'source.json.gz')
        angles=read(paths(entry)[1]/'angles.json')
        graph=evidence['scaffold']
        pic=Image.open(Path(entry['frame'])/'rgb.png').convert('RGB')
        draw=ImageDraw.Draw(pic)
        for a,b,axis in graph['edges']:
            line=[tuple(graph['xy'][k]) for k in (a,b)]
            draw.line(line,fill=['#ff9b2e','#67ffbd','#ff41cd'][axis],width=5)
        contact=None
        if evidence['tentative']:
            contact=evidence['tentative'][0]['contact']
            a,b=contact['edge']
            draw.line([tuple(graph['xy'][a]),tuple(graph['xy'][b])],fill='white',width=9)
            for k,col in [(a,'#5dfc8e'),(b,'#ff523b')]:
                x,y=graph['xy'][k]
                draw.ellipse((x-12,y-12,x+12,y+12),outline=col,width=5)
        pic=pic.resize((596,373),Image.Resampling.LANCZOS)
        x=600*i
        sheet.paste(pic,(x,36))
        pen.text((x+8,5),label,font=heading,fill='white')
        texts=(['Three colored axes trace observed spans.','Worst axis error 1.97 degrees, focal length 7.5% low.']
               if i==0 else ['White marks the measured wall base. Green marks a turning corner.','The open red endpoint leaves its length uncertain (3.71 blocks).'])
        for j,line in enumerate(texts):
            pen.text((x+8,415+j*23),line,font=font,fill='#e7ecf2')
        data.append(dict(id=key,source_sha256=entry['rgb_sha256'],angles=angles,contact=contact))
    sheet.save(REPORT/'castle-and-pillars.jpg',quality=92)
    save(REPORT/'illustrated-cases.json',data)


if __name__=='__main__':
    main()
